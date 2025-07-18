# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from typing import Deque, FrozenSet, Iterable, List, Optional, Tuple, Union

from vllm.core.block.common import (BlockPool, CopyOnWriteTracker, RefCounter,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import Block, BlockAllocator, BlockId, Device

Refcount = int


class NaiveBlockAllocator(BlockAllocator):
    """A simple block allocator that manages blocks of memory without prefix
    caching.

    Args:
        create_block (Block.Factory): A factory function for creating new
            blocks. This is used when a NaiveBlockAllocator is composed within
            a prefix caching allocator -- the naive block allocator must
            construct prefix caching blocks (but shouldn't know anything else
            about them).
        num_blocks (int): The total number of blocks to manage.
        block_size (int): The size of each block in tokens.
        block_ids (Optional[Iterable[int]], optional): An optional iterable of
            block IDs. If not provided, block IDs will be assigned sequentially
            from 0 to num_blocks - 1.
    """
    # 创建时分配的block是物理块的记录信息
    # block_pool则是逻辑块池
    # 允许block_pool 多余物理块的原因： 请求比较多时，可以先占用逻辑块暂存，当有物理块释放时，直接立即占用即可，
    # 从而避免同时等待逻辑块和物理块的释放。 相当于一个等待排队的队列，先准备好，然后随时执行，而不是等到可以执行时，开始准备。
    # 更核心的原因： 可能会有多个逻辑块指向物理块- 以便支持前缀缓存和更复杂的物理块共享机制

    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
        block_pool: Optional[BlockPool] = None,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)
        # 释放的block 块索引-非空的，所有可用block_ids
        self._free_block_indices: Deque[BlockId] = deque(block_ids)
        # 所有块索引- 将其转换成不可变的ids集合- 确保block_ids 和 num_blocks一致，那为什么不直接传入block集合呢？ 那可以避免这种不一致的风险了。
        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks
        # 引用计数： 
        self._refcounter = RefCounter(
            all_block_indices=self._free_block_indices)
        self._block_size = block_size
        # 将对象转换成只读对象，避免外部修改
        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly())

# block_pool 池？ 预分配超过当前num_blocks？ 那如何与blockId对应呢？
        if block_pool is None:
            extra_factor = 4
            # Pre-allocate "num_blocks * extra_factor" block objects.
            # The "* extra_factor" is a buffer to allow more block objects
            # than physical blocks - -为什么create_block要这么快指定？
            self._block_pool = BlockPool(self._block_size, create_block, self,
                                         num_blocks * extra_factor)
        else:
            # In this case, the block pool is provided by the caller,
            # which means that there is most likely a need to share
            # a block pool between allocators
            self._block_pool = block_pool

    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 extra_hash: Optional[int] = None,
                                 device: Optional[Device] = None) -> Block:
        """Allocates a new immutable block with the given token IDs, linked to
        the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.
            token_ids (List[int]): The token IDs to be stored in the new block.

        Returns:
            Block: The newly allocated immutable block.
        """ 
        # 为什么要asset device is None?  对于Immutable 其实底层也是mutableBlock
        assert device is None
        
        block_id = self._allocate_block_id()
        block = self._block_pool.init_block(prev_block=prev_block, 
                                            token_ids= token_ids,
                                            block_size=self._block_size,
                                            physical_block_id=block_id)
    
        return block

    def allocate_immutable_blocks(
            self,
            prev_block: Optional[Block],
            block_token_ids: List[List[int]],
            extra_hash: Optional[int] = None,
            device: Optional[Device] = None) -> List[Block]:
        assert device is None
        num_blocks = len(block_token_ids)
        # 批量分配时，提前预校验是否足够，不够直接返回，否则会导致内部数据错误
        if self.get_num_free_blocks() < num_blocks:
            raise BlockAllocator.NoFreeBlocksError()

        block_ids = []
        for i in range(num_blocks):
            block_ids.append(self._allocate_block_id())

        blocks = []
        for i in range(num_blocks):
            prev_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block_token_ids[i],
                block_size=self._block_size,
                physical_block_id=block_ids[i])
            blocks.append(prev_block)

        return blocks

    def allocate_mutable_block(self,
                               prev_block: Optional[Block], 
                               extra_hash: Optional[int] = None,
                               device: Optional[Device] = None) -> Block:
        """Allocates a new mutable block, linked to the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.

        Returns:
            Block: The newly allocated mutable block.
        """
        assert device is None
        # 分配物理blockId
        block_id = self._allocate_block_id()
        block = self._block_pool.init_block(prev_block=prev_block, 
                                            block_size=self._block_size,
                                            physical_block_id=block_id)
        # 返回逻辑block
        return block

    def _allocate_block_id(self) -> BlockId:
        if not self._free_block_indices:
            raise BlockAllocator.NoFreeBlocksError()
        # 为什么不将取操作和释放操作封装- 已经是在allocate内部了，这其实就是封装了
        block_id = self._free_block_indices.popleft()
        self._refcounter.incr(block_id)
        return block_id

    def _free_block_id(self, block: Union[Block, BlockId]) -> None:
        if isinstance(block, Block):
            block_id = block.block_id
            block.block_id = None
        else:
            block_id = block
        assert block_id is not None
        # 避免释放了并没有被分配的blockId，if block_id in _refcounter and in _free_block_indices,it could be error
        refcount = self._refcounter.decr(block_id)
        # 引用数为0时，重新加入到free集合中，这里应该校验一下 _free_block_indices中一定不包含此blockId,
        # 如果没有被引用，那么上一步就会报错
        if refcount == 0:
            # 为了代码健壮性，避免计算和分配不一致，可以加上校验。
            assert block_id not in self._free_block_indices
            self._free_block_indices.appendleft(block_id)

    def free(self, block: Block, keep_block_object: bool = False) -> None:
        # Release the physical block id
        self._free_block_id(block)

        # Release the block object
        if not keep_block_object:
            self._block_pool.free_block(block)

    def free_block_id(self, block_id: BlockId) -> None:
        self._free_block_id(block_id)

    # 创建一个新的序列块，与之前的块共享底层内存
    # 所以，会对底层内存添加引用，代表有多个逻辑块共同引用了底层内存，然后当需要写时，就copy一份单独的内存去写
    # 为什么要将前面所有的计数都加1？ 而不是仅增加前一个的引用计数呢？
    # 因为一个序列（sequence）通常由一系列逻辑块组成，这些逻辑块之间通过preBlock进行连接，同时这些preBlock正常对应的物理块也是不一样的？所以需要遍历进行
    # 序列之间会不会也有共享的block？ 有也没关系，释放时多次释放即可。
    def fork(self, last_block: Block) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
        memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: The new sequence of blocks that shares the same memory
                as the original sequence.
        """
        source_blocks = get_all_blocks_recursively(last_block)

        forked_blocks: List[Block] = []
        prev_block = None
        for block in source_blocks:

            # Increment refcount for each block.
            assert block.block_id is not None
            # block1引用了2，blcok1没有从中获取时，此时 refCount = 1, 只要有被引用过，就不会存在于free队列中了
            # 有可能后面其他适用了之前的block_id，之前的block会被分配给其他？然后block的引用不为空，于是会覆盖？
            assert self._refcounter.get(block.block_id) > 0, "can't fork free'd block"
            refcount = self._refcounter.incr(block.block_id)
            # 不能对free block进行fork，因为这样的block没有人引用，可以直接使用，而不是fork
            assert refcount != 1, "can't fork free'd block"

            forked_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block.token_ids,
                block_size=self._block_size,
                physical_block_id=block.block_id)

            forked_blocks.append(forked_block)
            prev_block = forked_blocks[-1]

        return forked_blocks

    def get_num_free_blocks(self) -> int:
        return len(self._free_block_indices)

    def get_num_total_blocks(self) -> int:
        return len(self._all_block_indices)

    def get_physical_block_id(self, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain block allocator
        given the absolute block id.

        Args:
            absolute_id (int): The absolute block id for the block 
            in whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return sorted(self._all_block_indices).index(absolute_id)

    @property
    def refcounter(self):
        return self._refcounter

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return self._all_block_indices

    def cow_block_if_not_appendable(self, block: Block) -> BlockId:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            BlockId: The block index of the new block if a copy-on-write 
                operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        src_block_id = block.block_id
        assert src_block_id is not None
        # 校验目标物理block是否独占，如果是共享的代表不能被追加，需要cow
        if self._cow_tracker.is_appendable(block):
            return src_block_id
        # 释放之前，是不是也应该先判断是否足够，够的时候，才进行，否则这个block的引用也被释放了
        if self.get_num_free_blocks() < 1:
            raise BlockAllocator.NoFreeBlocksError()
        # 释放当前块的引用，创建一个新的block
        self._free_block_id(block)
        trg_block_id = self._allocate_block_id()

        self._cow_tracker.record_cow(src_block_id, trg_block_id)
        # 返回新的blockId 
        return trg_block_id

    def clear_copy_on_writes(self) -> List[Tuple[BlockId, BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices.
        """
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as computed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        """Determine blocks that can be skipped in prefill.

        Since the naive allocator does not support prefix caching, always return
        an empty list.
        """
        return []

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        raise NotImplementedError("There is no promotion for naive blocks")

    def get_num_full_blocks_touched(self, blocks: List[Block]) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out.

        Args:
            blocks: List of blocks to be swapped.
        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks. Non full blocks are ignored
                when deciding the number of blocks to touch.
        """
        # NOTE: for naive block, we use set to eliminate common blocks among
        # seqs, also we compare the empty slots in the mutable blocks with
        # lookahead slots to get the number of unique new block that are
        # needed.
        old_block_set = set()
        for block in blocks:
            if block.is_full:
                old_block_set.add(block)
        return len(old_block_set)
# 仅仅为了获取物理blockId， 是呀，这样对于GPU——allocator而言就又有新的物理block可以使用了。
# 那物理blockId对应的物理块中存放的kvcache呢？ 其实还会存在，但这其实属于了无效数据，下一次使用时，会直接被覆盖。
# 所以换出之后，其实对应的缓存也会清理，这也很正常，反正又不会影响最终真实的数据，影响的仅仅是缓存而已。
    def swap_out(self, blocks: List[Block]) -> None:
        for block in blocks:
            self._free_block_id(block)
     
    def swap_in(self, blocks: List[Block]) -> None:
        for block in blocks:
            # Here we allocate either immutable or mutable block and then
            # extract its block_id. Note that the block object is released
            # and the block_id is assigned to "block" to allow reusing the
            # existing "block" object
            if block.is_full:
                tmp_block = self.allocate_immutable_block(
                    prev_block=block.prev_block, token_ids=block.token_ids)
            else:
                tmp_block = self.allocate_mutable_block(
                    prev_block=block.prev_block)
                tmp_block.append_token_ids(block.token_ids)

            block_id = tmp_block.block_id
            tmp_block.block_id = None
            self._block_pool.free_block(tmp_block)

            block.block_id = block_id  # Assign block_id

    def get_prefix_cache_hit_rate(self) -> float:
        return -1

    def reset_prefix_cache(self) -> bool:
        """No prefix cache for naive block allocator."""
        return True

    def find_cached_blocks_prefix(self, block_hashes: List[int]) -> List[int]:
        # Not applicable for naive block allocator.
        return []


class NaiveBlock(Block): 
    """An implementation of the Block class that does not support prefix
    caching.

    The NaiveBlock class represents a block of token IDs with a fixed size. It
    provides methods for appending token IDs to the block and manages copy-on
    -write operations when necessary.

    Args:
        prev_block (Block): The previous block in the sequence.
        token_ids (List[int]): The initial token IDs to be stored in the block.
        block_size (int): The maximum number of token IDs that can be stored in
            the block.
        allocator (BlockAllocator): The block allocator associated with this
            block.
        block_id (Optional[int], optional): The physical block index
            of this block. Defaults to None, which means no allocation has been
            made.
        _cow_target (Optional[Block], optional): The copy-on-write target block.
            If not provided, it defaults to self.
    """

    def __init__(self,
                 prev_block: Optional[Block],
                 token_ids: List[int],
                 block_size: int,
                 allocator: BlockAllocator,
                 block_id: Optional[int] = None,
                 _cow_target: Optional[Block] = None,
                 extra_hash: Optional[int] = None,
                 read_only: bool = False):
        self._token_ids: List[int] = []
        self._block_size = block_size
        self._prev_block = prev_block
        self._block_id = block_id
        self._allocator = allocator
        self._cow_target = _cow_target if _cow_target is not None else self
        self._read_only = read_only

        self._append_token_ids_no_cow(token_ids)
    # 将token_id 添加到token集合中, 会判断当前被添加的block是否是共享block，如果是，则需要进行cow
    def append_token_ids(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block and performs a 
        copy-on-write if necessary.

        Args:
            token_ids (Optional[List[int]]): The token IDs to be appended 
                to the block.
        """
        if self._read_only:
            raise ValueError("The block only read") # 写入一个错误类抛出
        self._append_token_ids_no_cow(token_ids)
        # 物理block不为空时，会判断是否进行copy-On-Write, 基于这个物理block是否被多个同时共享了。
        if self._block_id is not None:
            self._block_id = (self._allocator.cow_block_if_not_appendable(
                self._cow_target))

    def _append_token_ids_no_cow(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block

        Args:
            token_ids (List[int]): The token IDs to be appended to the block.
        """
        if token_ids is None:
            return
        if len(token_ids) == 0:
            return

        assert len(token_ids) <= self.num_empty_slots

        self._token_ids.extend(token_ids)

    @property
    def computed(self) -> bool:
        raise NotImplementedError

    @computed.setter
    def computed(self, value) -> None:
        raise NotImplementedError

    @property
    def last_accessed(self) -> float:
        raise NotImplementedError

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        raise NotImplementedError

    @property
    def block_id(self) -> Optional[int]:
        return self._block_id

    @block_id.setter
    def block_id(self, value: Optional[int]) -> None:
        self._block_id = value

    @property
    def is_full(self) -> bool:
        return self.num_empty_slots == 0

    @property
    def num_empty_slots(self) -> int:
        return self._block_size - len(self.token_ids)

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError(
            "num_tokens_total is not used for naive block")

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def prev_block(self) -> Optional["Block"]:
        return self._prev_block

    @property
    def extra_hash(self):
        return None

    @property
    def content_hash(self) -> Optional[int]:
        return None
