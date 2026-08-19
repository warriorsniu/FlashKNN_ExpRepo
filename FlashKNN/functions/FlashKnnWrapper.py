import time

import torch
import torch.nn as nn

from .CuFun import (
    FlashKNN_Back_Query_DL,
    FlashKNN_Morton_Encode,
    FlashKNN_Morton_Neighbor_Keys,
    FlashKNN_Nearest_Back_Query_DL,
    FlashKNN_Query_Dynamic_Load,
    FlashKNN_Query_Dynamic_Load_Ablation,
    FlashKNN_Query_Dynamic_Load_Thread_Group,
    FlashKNN_Query_GM,
    FlashKNN_Query_GMPS,
    FlashKNN_Query_SMSS,
    FlashKNN_Selected_Query_DL,
    get_cuda_shared_mem,
)
from .z_order import xyz2key

class FlashKNN():
    def __init__(
        self,
        num_nbr=32,
        num_down=2,
        debug = False,
        print_time = False,
        device = "cuda",
        multilevel_neigh = False,
        threshold = 256) -> None:
        # assert num_nbr <= 32, "邻域数量暂不支持超过32"
        # assert num_down == 2, "降采样次数建议为2"
        assert device == "cuda", "仅支持GPU"
        self.num_nbr = num_nbr
        self.num_down = num_down
        self.debug = debug
        self.print_time = print_time
        self.CudaShareMem = get_cuda_shared_mem()
        self.threshold = threshold
        if self.debug:
            self.time_list = []

        if multilevel_neigh:
            neigh_grid_array = torch.tensor([0,-1,1], device=device, dtype=torch.long)
            self.grid_offset = self.rng_grid(neigh_grid_array,device)
            oct_offset = self.rng_grid([0,1], device)

            child_offset = oct_offset[:,None,:] + self.grid_offset[None,:,:]
            parent_offset = torch.floor((child_offset) / 2).long()    #8*27*3
            child_offset = child_offset&1
            parent_neigh_offset_key = ((self.grid_offset + 1) * torch.tensor([9,3,1], device=device, dtype=torch.long)).sum(dim=1) 
            parent_neigh_offset_order = torch.zeros_like(parent_neigh_offset_key)
            parent_neigh_offset_order[parent_neigh_offset_key] = torch.arange(len(parent_neigh_offset_key), device=device)

            child_neigh_offset_key = ((oct_offset) * torch.tensor([4,2,1], device=device, dtype=torch.long)).sum(dim=1) 
            child_neigh_offset_order = torch.zeros_like(child_neigh_offset_key)
            child_neigh_offset_order[child_neigh_offset_key] = torch.arange(len(child_neigh_offset_key), device=device)

            parent_neigh_key = ((parent_offset + 1) * torch.tensor([9,3,1], device=device, dtype=torch.long)).sum(dim=2) 
            child_neigh_key = ((child_offset) * torch.tensor([4,2,1], device=device, dtype=torch.long)).sum(dim=2) 

            self.neigh_lut_parent = parent_neigh_offset_order[parent_neigh_key]
            self.neigh_lut_child = child_neigh_offset_order[child_neigh_key]

        
    def construct_neigh():
        pass
    
    def construct_multilevel_neigh(
            self, 
            grid_coord:torch.Tensor, 
            batch_idx:torch.Tensor,
            depth = 4):
        ts = time.time()
        device = grid_coord.device
        grid_offset = self.grid_offset

        key = xyz2key(grid_coord[:,0], grid_coord[:,1], grid_coord[:,2], batch_idx)
        order = torch.argsort(key)
        key = key[order]
        grid_coord = grid_coord[order]
        inverse_order = torch.zeros(len(grid_coord), device=device, dtype=torch.long)
        inverse_order[order] = torch.arange(len(grid_coord), device=device, dtype=torch.long)

        key_down = key >> (3*self.num_down)
        grid_coord_down = grid_coord >> (self.num_down)
        key_down_unique, Child2Parent, count = torch.unique_consecutive(key_down, return_inverse = True, return_counts=True)
        steps = torch.cumsum(count, dim=0)
        steps = nn.functional.pad(steps, (1,0))
        grid_coord_down = grid_coord_down[steps[:len(key_down_unique)]]

        WE2NE_list = []
        neigh_list = []
        steps_list = [steps]
        count_list = [count]


        key_child = key_down_unique
        grid_coord_child = grid_coord_down
        count_child = count
        for _ in range(depth-1):
            key_parent = key_child >> 3
            grid_coord_parent = grid_coord_child >> 1
            res = grid_coord_child&1
            offset = (res[:,2] << 2)|(res[:,1] << 1)|(res[:,0] << 0)
            key_parent_unique, Child2Parent_leveli, count = torch.unique_consecutive(
                key_parent, return_inverse = True, return_counts=True)
            WE2NE = -key_parent_unique.new_ones(len(key_parent_unique)*8)
            WE2NE[Child2Parent_leveli*8 + offset] = torch.arange(len(key_child), device=device)   #key 和 grid是否一致？
            WE2NE_list.append(WE2NE)
            steps_lv_i = nn.functional.pad(torch.cumsum(count, dim=0), (1,0))
            count_lvi_2_lv0 = torch.zeros(len(count), dtype=torch.long, device=device)
            count_lvi_2_lv0 = count_lvi_2_lv0.scatter_add(dim=0, index=Child2Parent_leveli, src=count_child)
            steps_lvi_2_lv0 = nn.functional.pad(torch.cumsum(count_lvi_2_lv0, dim=0), (1,0))
            grid_coord_child = grid_coord_parent[steps_lv_i[:len(key_parent_unique)]]
            key_child = key_parent[steps_lv_i[:len(key_parent_unique)]]
            count_child = count_lvi_2_lv0
            count_list.append(count_child)
            steps_list.append(steps_lvi_2_lv0)
        
        if self.debug:
            print("降采样耗时：", time.time() - ts);ts = time.time()
        # 构建最底层邻域图
        grid_coord_bottom_nbr = (grid_coord_child.unsqueeze(1) + grid_offset.unsqueeze(0)).view(-1,3)       # (N*27, 3)
        key_bottom_nbr = xyz2key(
            grid_coord_bottom_nbr[:,0], 
            grid_coord_bottom_nbr[:,1], 
            grid_coord_bottom_nbr[:,2], 
            None if batch_idx is None else batch_idx[steps_list[-1][:len(grid_coord_child)]].unsqueeze(1).repeat((1,27)).view(-1))

        key_bottom_unique_nbr, inverse = torch.unique(torch.cat([key_bottom_nbr[::27], key_bottom_nbr]), return_inverse = True)

        valid_key_down_index = inverse[:len(grid_coord_child)]
        num_unique_keys = len(key_bottom_unique_nbr)

        lut = -torch.ones(num_unique_keys, dtype = torch.long, device=device)
        lut[valid_key_down_index] = torch.arange(len(grid_coord_child), dtype = torch.long, device=device)
        neigh = lut[inverse[len(grid_coord_child):]].reshape(-1, 27)
        neigh_list.insert(0, neigh)

        if self.debug:
            print("底层邻域图构建耗时：", time.time() - ts);ts = time.time()
        
        for lv in  reversed(range(depth-1)):
            parent_neigh = neigh_list[0]
            parent_neigh = parent_neigh[:, self.neigh_lut_parent]
            child_neigh = parent_neigh*8 + self.neigh_lut_child[None,:,:]
            WE2NE = WE2NE_list[lv]
            NE_mask = WE2NE >= 0
            invalid_mask = (child_neigh < 0) | (parent_neigh < 0)
            child_neigh = WE2NE[child_neigh]
            child_neigh[invalid_mask] = -1
            child_neigh = child_neigh.view(-1,27)[NE_mask]
            neigh_list.insert(0, child_neigh)

        self.neigh_list = neigh_list
        self.count_list = count_list
        self.steps_list = steps_list

        if self.debug:
            print("全部邻域图构建耗时：", time.time() - ts);ts = time.time()

        
    @torch.no_grad()
    def query(
            self, 
            grid_coord:torch.Tensor, 
            batch_idx:torch.Tensor,
            xyz:torch.Tensor = None,
            dynamic_load = True,
            cut_radius = torch.inf,
            batch_for_prune = 1,
            memory_mode = "SM",
            sorting_mode = "PS",
            candidate_mode = "register",
            enable_skip = True,
            thread_group_size = None,
            traverse_info = None,
            ):
        """
        grid_coord:         网格坐标
        batch_idx:  
        xyz:                原始坐标
        dynamic_load:       在使用共享内存时, 是否动态读取support点
        cut_radius:         截断距离
        batch_for_prune:    剪枝批次
        memory_mode:        内存模式, SM:共享内存, GM: 全局内存, Hybrid: 混合内存
        candidate_mode:     PS候选列表位置, register 或 shared
        enable_skip:        PS是否跳过不含更优候选的排序轮次
        thread_group_size:  None为生产自适应策略，或固定为8/16/32（仅用于消融）
        """
        ts = time.time()
        ts_origin = time.time()
        device = grid_coord.device
        if candidate_mode not in {"register", "shared"}:
            raise ValueError(
                f"candidate_mode must be 'register' or 'shared', got {candidate_mode!r}"
            )
        if thread_group_size not in {None, 8, 16, 32}:
            raise ValueError(
                "thread_group_size must be None, 8, 16, or 32, got "
                f"{thread_group_size!r}"
            )
        if (candidate_mode != "register" or not enable_skip) and not (
            memory_mode == "SM" and sorting_mode == "PS"
        ):
            raise ValueError(
                "candidate placement/skip ablations require memory_mode='SM' "
                "and sorting_mode='PS'"
            )
        if thread_group_size is not None and not (
            memory_mode == "SM" and sorting_mode == "PS"
            and candidate_mode == "register" and enable_skip
        ):
            raise ValueError(
                "fixed thread-group ablations require memory_mode='SM', "
                "sorting_mode='PS', register candidates, and skip enabled"
            )
        fused_construction = (
            batch_idx is not None
            and grid_coord.is_cuda
            and grid_coord.dtype == torch.int64
            and batch_idx.dtype == torch.int64
        )
        if fused_construction:
            key = FlashKNN_Morton_Encode(
                grid_coord.contiguous(), batch_idx.contiguous()
            )
        else:
            grid_offset = self.rng_grid([0,-1,1],device)
            key = xyz2key(grid_coord[:,0], grid_coord[:,1], grid_coord[:,2], batch_idx)
        order = torch.argsort(key)
        key = key[order]
        grid_coord = grid_coord[order]
        inverse_order = torch.empty(len(grid_coord), device=device, dtype=torch.long)
        inverse_order[order] = torch.arange(len(grid_coord), device=device, dtype=torch.long)

        key_down = key >> (3*self.num_down)
        needs_child_to_parent = memory_mode == "GM" and sorting_mode == "SS"
        if needs_child_to_parent:
            key_down_unique, Child2Parent, count = torch.unique_consecutive(
                key_down, return_inverse=True, return_counts=True
            )
        else:
            key_down_unique, count = torch.unique_consecutive(
                key_down, return_counts=True
            )
            Child2Parent = None
        steps = torch.cumsum(count, dim=0)
        steps = nn.functional.pad(steps, (1,0))
        parent_starts = steps[:len(key_down_unique)]
        grid_coord_down = (grid_coord[parent_starts] >> self.num_down).contiguous()
        if self.debug:
            if self.print_time:
                print(f"预处理_降采样耗时: {time.time() - ts}")
            ts = time.time()

        if fused_construction:
            parent_batch = batch_idx[order[parent_starts]].contiguous()
            key_down_nbr = FlashKNN_Morton_Neighbor_Keys(
                grid_coord_down, parent_batch
            )
        else:
            grid_coord_down_nbr = (
                grid_coord_down.unsqueeze(1) + grid_offset.unsqueeze(0)
            ).view(-1,3)
            key_down_nbr = xyz2key(
                grid_coord_down_nbr[:,0],
                grid_coord_down_nbr[:,1],
                grid_coord_down_nbr[:,2],
                None if batch_idx is None else batch_idx[order[parent_starts]].unsqueeze(1).repeat((1,27)).view(-1))

        if self.debug:
            if self.print_time:
                print(f"预处理_邻域key计算耗时: {time.time() - ts}")
            ts = time.time()
            
        if fused_construction:
            parent_keys = key_down_nbr[::27].contiguous()
            positions = torch.searchsorted(parent_keys, key_down_nbr)
            safe_positions = positions.clamp_max(len(parent_keys) - 1)
            present = (
                (positions < len(parent_keys))
                & (parent_keys[safe_positions] == key_down_nbr)
            )
            neigh = torch.where(present, safe_positions, -1).reshape(-1, 27)
        else:
            key_down_unique_full_nbr, inverse = torch.unique(
                torch.cat([key_down_nbr[::27], key_down_nbr]),
                return_inverse=True,
            )

        # key_down_full_nbr_ordered, order_full_nbr = torch.cat([key_down_nbr[::27], key_down_nbr]).sort()
        # order_inverse_full_nbr = torch.zeros_like(order_full_nbr)
        # order_inverse_full_nbr[order_full_nbr] = torch.arange(len(order_inverse_full_nbr), device=order_inverse_full_nbr.device)
        # key_down_unique_full_nbr, inverse = torch.unique_consecutive(key_down_full_nbr_ordered, return_inverse = True)
        # inverse = inverse[order_inverse_full_nbr]

        if self.debug:
            if self.print_time:
                print(f"预处理_邻域图构建耗时: {time.time() - ts}")
            ts = time.time()

        if not fused_construction:
            valid_key_down_index = inverse[:len(key_down_unique)]
            num_unique_keys = len(key_down_unique_full_nbr)
            lut = -torch.ones(num_unique_keys, dtype = torch.long, device=device)
            lut[valid_key_down_index] = torch.arange(
                len(key_down_unique), dtype=torch.long, device=device
            )
            neigh = lut[inverse[len(key_down_unique):]].reshape(-1, 27)

        
        # 调用cuda kernel寻找邻域点
        """
        输入
        子节点坐标           grid_coord
        父节点查询子节点     steps
        子节点查询父节点     Child2Parent
        父节点3*3*3邻接图    neigh
        邻接图内候选累计数量 cnt_in_neigh
        邻接图内候选总数量   cnt_in_neigh[:,-1]
        输出
        
        """

        # def check_neigh_outofbound(neigh: torch.Tensor, coord: torch.Tensor):
        #     neigh_debug = neigh.clone()
        #     neigh_debug[neigh_debug == -1] = (torch.arange(len(neigh_debug), device=neigh.device)[:,None].repeat(1,27))[neigh_debug == -1]
        #     neigh_coord = grid_coord_down[neigh_debug]
        #     outofboundcheck = (neigh_coord - neigh_coord[:,0:1,:]).abs().max(dim=-1)[0]
        #     return outofboundcheck
        # outofboundcheck = check_neigh_outofbound(neigh, grid_coord_down)

        # child_idx = torch.arange(len(grid_coord), device=device)
        cnt_in_neigh = nn.functional.pad(torch.cumsum(nn.functional.pad(count, (0,1))[neigh], dim=1), [1,0])
        out_indices = torch.zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.int32)
        
        if self.debug:
            torch.cuda.synchronize()
            time_cost_structure_construct = time.time() - ts_origin
            if self.print_time:
                print(f"子节点数量: {len(grid_coord)}")
                print(f"父节点数量: {len(neigh)}")
                print(f"预处理耗时: {time_cost_structure_construct}")
            ts = time.time()
            
            # print("平均候选点数量: ", cnt_in_neigh[:,-1].float().mean())
            # print(cnt_in_neigh[:,-1].contiguous().int().min())
        if traverse_info is not None:
            traverse_info[0] = cnt_in_neigh[:,-1].float().mean().cpu().item()
        if xyz is None:
            intput_xyz = grid_coord.float()
        else:
            intput_xyz = xyz[order].float().contiguous()
        if memory_mode == "SM":
            out_dis = grid_coord.new_zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.float32)
            # print(cnt_in_neigh[14])
            if sorting_mode == "PS":
                arguments = (
                    intput_xyz,
                    steps.int(),
                    neigh.int(),
                    cnt_in_neigh.int(),
                    self.num_nbr,
                    out_indices,
                    out_dis,
                    batch_for_prune,
                    cut_radius*cut_radius,
                )
                if thread_group_size is not None:
                    FlashKNN_Query_Dynamic_Load_Thread_Group(
                        *arguments, thread_group_size,
                    )
                elif candidate_mode == "register" and enable_skip:
                    FlashKNN_Query_Dynamic_Load(*arguments)
                else:
                    FlashKNN_Query_Dynamic_Load_Ablation(
                        *arguments,
                        candidate_mode == "shared",
                        enable_skip,
                    )
            elif sorting_mode == "SS":
                FlashKNN_Query_SMSS(
                    intput_xyz, 
                    steps.int(),  
                    neigh.int(), 
                    cnt_in_neigh.int(), 
                    self.num_nbr,
                    out_indices,
                    out_dis,
                    batch_for_prune,
                    cut_radius*cut_radius
                )
        elif memory_mode == "GM":
            out_dis = grid_coord.new_zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.float32)
            if sorting_mode == "SS":
                assert Child2Parent is not None
                steps_int = steps.int()
                FlashKNN_Query_GM(
                    intput_xyz,
                    intput_xyz,
                    steps_int,
                    steps_int,
                    torch.arange(len(xyz), device=intput_xyz.device, dtype=torch.int32),
                    Child2Parent.int(),
                    neigh.int(), 
                    cnt_in_neigh.int(), 
                    out_indices,
                    out_dis,
                    self.num_nbr,
                    cut_radius*cut_radius
                )
            elif sorting_mode == "PS":
                FlashKNN_Query_GMPS(
                    intput_xyz, 
                    steps.int(),  
                    neigh.int(), 
                    cnt_in_neigh.int(), 
                    self.num_nbr,
                    out_indices,
                    out_dis,
                    batch_for_prune,
                    cut_radius*cut_radius
                )
        out_indices = out_indices.long()
        # print(out_indices[7319].sort()[0])
        out_indices = order[out_indices[inverse_order]]
        torch.cuda.synchronize()
        if self.debug:
            query_time = time.time() - ts
            if self.print_time:
                print(f"查询耗时: {query_time}")
            self.time_list.append(
                {
                    "预处理耗时": time_cost_structure_construct, 
                    "查询耗时": query_time, 
                    "查询类型": "query",
                    "memory_mode": memory_mode,
                    "sorting_mode": sorting_mode,
                    "candidate_mode": candidate_mode,
                    "enable_skip": enable_skip,
                    "thread_group_size": (
                        "adaptive" if thread_group_size is None
                        else thread_group_size
                    ),
                    "dynamic_load": dynamic_load})
        return out_indices
    
    def rng_grid(self, rng, device = "cuda"):
        r''' Builds a mesh grid in :obj:`[min, max]` (:attr:`max` included).
        '''
        if(isinstance(rng, list)):
            rng = torch.Tensor(rng).long().to(device)
        grid = torch.meshgrid(rng, rng, rng, indexing="ij")
        grid = torch.stack(grid, dim=-1).view(-1, 3)  # (27, 3)
        return grid
    
    def back_query(
            self,
            xyz:torch.Tensor,
            sel_indices:torch.Tensor, 
            query_grid_size:float,
            down_grid_size: float,
            batch_idx:torch.Tensor,
            dynamic_load = True,
            memory_mode = "SM"):
        """
        xyz:                原始坐标
        sel_indices:        降采样选择点(support点)
        query_grid_size:    执行查询的降采样层次
        down_grid_size:     降采样点所在层次的grid_size
        batch_idx:  
        dynamic_load:       在使用共享内存时, 是否动态读取support点
        memory_mode:        内存模式, SM:共享内存, GM: 全局内存
        """
        ts = time.time()
        device = xyz.device
        grid_coord = (xyz/query_grid_size).long()
        fused_construction = (
            batch_idx is not None
            and grid_coord.is_cuda
            and grid_coord.dtype == torch.int64
            and batch_idx.dtype == torch.int64
        )
        if fused_construction:
            key_h = FlashKNN_Morton_Encode(
                grid_coord.contiguous(), batch_idx.contiguous()
            )
        else:
            grid_offset = self.rng_grid([0,-1,1],device)
            key_h = xyz2key(
                grid_coord[:,0], grid_coord[:,1], grid_coord[:,2], batch_idx
            )
        order_h = key_h.argsort()
        inverse_order = torch.empty(len(order_h), device=device, dtype=torch.long)
        inverse_order[order_h] = torch.arange(len(order_h), device=device, dtype=torch.long)

        # 新版本
        key_l = key_h[sel_indices]
        order_l = key_l.argsort()
        xyz_l = xyz[sel_indices][order_l]  #低分辨率点云
        ##

        key_h = key_h[order_h]
        key_unique_h, Child2Parent_h, count_h = torch.unique_consecutive(key_h, return_inverse = True, return_counts=True)
        steps_h = nn.functional.pad(torch.cumsum(count_h, dim=0), (1,0))
        xyz_h = xyz[order_h]               #高分辨率点云

        #新版本
        count_l = torch.zeros_like(count_h)
        count_l = count_l.scatter_add(
            dim=0, 
            index=Child2Parent_h[inverse_order[sel_indices]], 
            src=torch.ones_like(sel_indices))
        steps_l = torch.cumsum(count_l, dim=0)
        steps_l = nn.functional.pad(steps_l, (1,0))
        ##

        # 旧版本
        # key_l = xyz2key(
        #     grid_coord[sel_indices,0], 
        #     grid_coord[sel_indices,1], 
        #     grid_coord[sel_indices,2], 
        #     None if batch_idx is None else batch_idx[sel_indices])
        
        # order_l = key_l.argsort()
        # key_l = key_l[order_l]
        # key_unique_l, Child2Parent_l, count_l = torch.unique_consecutive(key_l, return_inverse = True, return_counts=True)
        # steps_l = nn.functional.pad(torch.cumsum(count_l, dim=0), (1,0))
        # xyz_l = xyz[sel_indices][order_l]

        # assert len(key_unique_h) == len(key_unique_l), "降采样前后的点所占网格数应该相等"

        parent_starts = steps_h[:len(key_unique_h)]
        grid_coord_down = grid_coord[order_h[parent_starts]].contiguous()
        if fused_construction:
            parent_batch = batch_idx[order_h[parent_starts]].contiguous()
            key_down_nbr = FlashKNN_Morton_Neighbor_Keys(
                grid_coord_down, parent_batch
            )
            parent_keys = key_down_nbr[::27].contiguous()
            positions = torch.searchsorted(parent_keys, key_down_nbr)
            safe_positions = positions.clamp_max(len(parent_keys) - 1)
            present = (
                (positions < len(parent_keys))
                & (parent_keys[safe_positions] == key_down_nbr)
            )
            neigh = torch.where(present, safe_positions, -1).reshape(-1, 27)
        else:
            grid_coord_down_nbr = (
                grid_coord_down.unsqueeze(1) + grid_offset.unsqueeze(0)
            ).view(-1,3)
            key_down_nbr = xyz2key(
                grid_coord_down_nbr[:,0],
                grid_coord_down_nbr[:,1],
                grid_coord_down_nbr[:,2],
                None if batch_idx is None else batch_idx[order_h[parent_starts]].unsqueeze(1).repeat((1,27)).view(-1))
            key_down_unique_full_nbr, inverse = torch.unique(
                torch.cat([key_unique_h, key_down_nbr]), return_inverse=True
            )
            valid_key_down_index = inverse[:len(key_unique_h)]
            num_unique_keys = len(key_down_unique_full_nbr)
            lut = -torch.ones(num_unique_keys, dtype=torch.long, device=device)
            lut[valid_key_down_index] = torch.arange(
                len(key_unique_h), dtype=torch.long, device=device
            )
            neigh = lut[inverse[len(key_unique_h):]].reshape(-1, 27)
        # 候选点数量取决于grid_size， 当grid_size二倍于降采样grid_size时，候选点大约6^2个，最多6^3

        support_cnt_in_neigh = nn.functional.pad(torch.cumsum(nn.functional.pad(count_l, (0,1))[neigh], dim=1), [1,0])
        out_indices = torch.zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.int32)
        out_dis = torch.zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.float32)
        if self.debug:
            torch.cuda.synchronize()
            print("预处理耗时: ", time.time() - ts);ts = time.time()

        assert (support_cnt_in_neigh[:,-1] != 0).all(), "某些query点邻域内无support点，请考虑增大query_grid_size"
        if memory_mode == "SM":
            if self.num_nbr == 1:
                
                FlashKNN_Nearest_Back_Query_DL(
                    xyz_h.float(),
                    xyz_l.float(),
                    steps_h.int(),
                    steps_l.int(),
                    neigh.int(),
                    support_cnt_in_neigh.int(),
                    out_indices,
                    out_dis
                )
                out_indices = out_indices.flatten().long()
                torch.cuda.synchronize()
                if self.debug:
                    torch.cuda.synchronize()
                    print("查询耗时: ", time.time() - ts);ts = time.time()
                return order_l[out_indices[inverse_order]]
            else:
                FlashKNN_Back_Query_DL(
                    xyz_h.float(),
                    xyz_l.float(),
                    steps_h.int(),
                    steps_l.int(),
                    neigh.int(),
                    support_cnt_in_neigh.int(),
                    self.num_nbr,
                    out_indices,
                    out_dis
                )
                # raise NotImplementedError
                out_indices = out_indices.long()
                out_indices = order_l[out_indices[inverse_order]]
                out_dis = out_dis[inverse_order]
                torch.cuda.synchronize()
                if self.debug:
                    torch.cuda.synchronize()
                    print("查询耗时: ", time.time() - ts);ts = time.time()
                return out_indices, out_dis
        elif memory_mode == "GM":
            FlashKNN_Query_GM(
                xyz_l.float(),
                xyz_h.float(),
                steps_l.int(),
                steps_h.int(),
                torch.zeros(len(xyz_h), device=xyz_h.device, dtype=torch.int32), #默认query点映射回support的索引0
                Child2Parent_h.int(),
                neigh.int(),
                support_cnt_in_neigh.int(),
                out_indices,
                out_dis,
                self.num_nbr,
                torch.inf
            )
            out_indices = out_indices.long()
            out_indices = order_l[out_indices[inverse_order]]
            out_dis = out_dis[inverse_order]
            if self.debug:
                torch.cuda.synchronize()
                print("查询耗时: ", time.time() - ts);ts = time.time()
            if self.num_nbr == 1:
                return out_indices
            else:
                return out_indices, out_dis

    @torch.no_grad()
    def selected_query(
        self,
        xyz:torch.Tensor,
        grid_coord:torch.Tensor,
        query_indices:torch.Tensor, 
        batch_idx:torch.Tensor,
        dynamic_load = True,
        cut_radius = torch.inf,
        batch_for_prune = 1,
        memory_mode = "SM"
    ):
        """
        xyz:                原始坐标
        grid_coord:         体素化坐标
        query_indices:      降采样选择点(query点)
        batch_idx:  
        dynamic_load:       在使用共享内存时, 是否动态读取support点
        cut_radius:         查询的截断距离
        batch_for_prune:    剪枝批次
        memory_mode:        内存模式, SM:共享内存, GM: 全局内存
        """
        ts = time.time()
        ts_origin = time.time()
        device = grid_coord.device
        fused_construction = (
            batch_idx is not None
            and grid_coord.is_cuda
            and grid_coord.dtype == torch.int64
            and batch_idx.dtype == torch.int64
        )
        if fused_construction:
            key_support = FlashKNN_Morton_Encode(
                grid_coord.contiguous(), batch_idx.contiguous()
            )
        else:
            grid_offset = self.rng_grid([0,-1,1],device)
            key_support = xyz2key(
                grid_coord[:,0], grid_coord[:,1], grid_coord[:,2], batch_idx
            )
        key_query = key_support[query_indices]
        order_support = torch.argsort(key_support)
        order_query = torch.argsort(key_query)

        key_support = key_support[order_support]
        grid_coord = grid_coord[order_support]
        inverse_order_support = torch.empty(len(grid_coord), device=device, dtype=torch.long)
        inverse_order_support[order_support] = torch.arange(len(grid_coord), device=device, dtype=torch.long)

        inverse_order_query = torch.empty(len(query_indices), device=device, dtype=torch.long)
        inverse_order_query[order_query] = torch.arange(len(query_indices), device=device, dtype=torch.long)

        key_down = key_support >> (3*self.num_down)
        key_down_unique, count = torch.unique_consecutive(
            key_down, return_counts=True
        )
        steps = torch.cumsum(count, dim=0)
        steps = nn.functional.pad(steps, (1,0))
        parent_starts = steps[:len(key_down_unique)]
        grid_coord_down = (grid_coord[parent_starts] >> self.num_down).contiguous()

        query_parent = torch.searchsorted(
            key_down_unique, (key_query >> (3*self.num_down)).contiguous()
        )
        count_query = torch.zeros_like(count)
        count_query = count_query.scatter_add(
            dim=0,
            index=query_parent,
            src=torch.ones_like(query_indices))
        steps_query = torch.cumsum(count_query, dim=0)
        steps_query = nn.functional.pad(steps_query, (1,0))

        if self.debug:
            time_down = time.time() - ts;ts = time.time()
            if self.print_time:
                print(f"预处理_降采样耗时: {time_down}")

        if fused_construction:
            parent_batch = batch_idx[order_support[parent_starts]].contiguous()
            key_down_nbr = FlashKNN_Morton_Neighbor_Keys(
                grid_coord_down, parent_batch
            )
        else:
            grid_coord_down_nbr = (
                grid_coord_down.unsqueeze(1) + grid_offset.unsqueeze(0)
            ).view(-1,3)
            key_down_nbr = xyz2key(
                grid_coord_down_nbr[:,0],
                grid_coord_down_nbr[:,1],
                grid_coord_down_nbr[:,2],
                None if batch_idx is None else batch_idx[order_support[parent_starts]].unsqueeze(1).repeat((1,27)).view(-1))

        if self.debug:
            time_27nbr_key = time.time() - ts;ts = time.time()
            if self.print_time:
                print(f"预处理_邻域key计算耗时: {time_27nbr_key}")

        if fused_construction:
            parent_keys = key_down_nbr[::27].contiguous()
            positions = torch.searchsorted(parent_keys, key_down_nbr)
            safe_positions = positions.clamp_max(len(parent_keys) - 1)
            present = (
                (positions < len(parent_keys))
                & (parent_keys[safe_positions] == key_down_nbr)
            )
            neigh = torch.where(present, safe_positions, -1).reshape(-1, 27)
        else:
            key_down_unique_full_nbr, inverse = torch.unique(
                torch.cat([key_down_nbr[::27], key_down_nbr]),
                return_inverse=True,
            )

        # key_down_full_nbr_ordered, order_full_nbr = torch.cat([key_down_nbr[::27], key_down_nbr]).sort()
        # order_inverse_full_nbr = torch.zeros_like(order_full_nbr)
        # order_inverse_full_nbr[order_full_nbr] = torch.arange(len(order_inverse_full_nbr), device=order_inverse_full_nbr.device)
        # key_down_unique_full_nbr, inverse = torch.unique_consecutive(key_down_full_nbr_ordered, return_inverse = True)
        # inverse = inverse[order_inverse_full_nbr]

        
        if not fused_construction:
            valid_key_down_index = inverse[:len(key_down_unique)]
            num_unique_keys = len(key_down_unique_full_nbr)
            lut = -torch.ones(num_unique_keys, dtype=torch.long, device=device)
            lut[valid_key_down_index] = torch.arange(
                len(key_down_unique), dtype=torch.long, device=device
            )
            neigh = lut[inverse[len(key_down_unique):]].reshape(-1, 27)
        
        # 调用cuda kernel寻找邻域点
        """
        输入
        子节点坐标           grid_coord
        父节点查询子节点     steps
        子节点查询父节点     Child2Parent
        父节点3*3*3邻接图    neigh
        邻接图内候选累计数量 cnt_in_neigh
        邻接图内候选总数量   cnt_in_neigh[:,-1]
        输出
        
        """
        # child_idx = torch.arange(len(grid_coord), device=device)
        cnt_in_neigh = nn.functional.pad(torch.cumsum(nn.functional.pad(count, (0,1))[neigh], dim=1), [1,0])
        out_indices = torch.zeros((len(query_indices), self.num_nbr), device=device, dtype=torch.int32)
        if self.debug:
            torch.cuda.synchronize()
            time_cost_structure_construct = time.time() - ts_origin
            time_nbr_neigh = time.time() - ts;ts = time.time()
            if self.print_time:
                print(f"预处理_邻域图构建耗时: {time_nbr_neigh}")
                print(f"预处理耗时: {time_cost_structure_construct}")

        if xyz is None:
            intput_xyz = grid_coord.float().contiguous()
        else:
            intput_xyz = xyz[order_support].float().contiguous()
            # out_dis = grid_coord.new_zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.int32)
        if memory_mode == "SM":
            out_dis = grid_coord.new_zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.float32)
            FlashKNN_Selected_Query_DL(
                intput_xyz,
                inverse_order_support[query_indices[order_query]].int().contiguous(),
                steps.int(),
                steps_query.int(),
                neigh.int(),
                cnt_in_neigh.int(),
                self.num_nbr,
                out_indices,
                out_dis,
                batch_for_prune,
                cut_radius*cut_radius
            )
        elif memory_mode == "GM":
            out_dis = grid_coord.new_zeros((len(grid_coord), self.num_nbr), device=device, dtype=torch.float32)
            QueryIndex = inverse_order_support[query_indices[order_query]].contiguous()
            FlashKNN_Query_GM(
                intput_xyz,
                intput_xyz[QueryIndex].contiguous(),
                steps.int(),
                steps_query.int(),
                QueryIndex.int(),
                query_parent[order_query].int().contiguous(),
                neigh.int(),
                cnt_in_neigh.int(),
                out_indices,
                out_dis,
                self.num_nbr,
                cut_radius*cut_radius
            )
        out_indices = out_indices.long()
        out_indices = order_support[out_indices[inverse_order_query]]
        torch.cuda.synchronize()
        if self.debug:
            time_query = time.time() - ts
            if self.print_time:
                print(f"查询耗时: {time_query}")
            self.time_list.append(
                {
                    "预处理耗时": time_cost_structure_construct, 
                    "查询耗时": time_query, 
                    "查询类型": "selected_query",
                    "memory_mode": memory_mode,
                    "dynamic_load": dynamic_load
                    })
        return out_indices
    
