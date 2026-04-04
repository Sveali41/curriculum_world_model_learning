import torch
from torch import nn
from torch import nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return x + self.fc2(self.dropout(self.relu(self.fc1(x))))

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        :param d_model: 特征维度
        :param nhead: 注意力头数
        :param dropout: dropout 比例
        """
        super(CustomTransformerEncoderLayer, self).__init__()
        # 使用 nn.MultiheadAttention，batch_first=True 便于使用形状 (B, seq_len, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        # 两个 LayerNorm 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 输入张量，形状 (B, seq_len, d_model)
        :return:
            - src: Transformer Encoder 层输出 (B, seq_len, d_model)
            - attn_weights: 注意力权重，形状 (B, num_heads, seq_len, seq_len)
        """
        # 计算自注意力，并返回注意力权重
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        # 残差连接 + LayerNorm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # 前馈网络部分
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, attn_weights


# class AttentionModule(nn.Module):
#     def __init__(self, data_type, grid_shape, mask_size, embed_dim, num_heads):
#         super().__init__()
#         self.data_type = data_type
#         if data_type == 'discrete':
#             self.input_channel = 21
#             self.action_embedding = nn.Embedding(5, embed_dim)
#             self.key_embedding    = nn.Embedding(2, embed_dim)
#         else:
#             self.input_channel = grid_shape[0]
#             self.action_fc = nn.Linear(1, embed_dim)
 
#         self.mask_size = mask_size
#         self.y, self.x = mask_size // 2, mask_size // 2
#         self.conv1 = nn.Conv2d(self.input_channel, embed_dim, kernel_size=3, padding=1)
#         self.bn1 = nn.GroupNorm(8, embed_dim)
#         self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
#         self.bn2 = nn.GroupNorm(8, embed_dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.to_gamma_beta = nn.Linear(embed_dim, 2 * embed_dim)

#         # 2. 展平操作，将 (B, embed_dim, H, W) 展平为 (B, embed_dim, H*W)
#         self.flatten = nn.Flatten(2)
#         # 位置编码：为每个 patch 学习一个位置编码，形状为 (1, height*width, embed_dim)
#         # self.pos_embedding = nn.Parameter(torch.randn(1, mask_size * mask_size, embed_dim))
#         self.pos_embedding = nn.Parameter(torch.zeros(1, mask_size * mask_size, embed_dim))
#         nn.init.trunc_normal_(self.pos_embedding, std=0.02)  # 更稳的初始化

#         # 3. 动作嵌入：将离散动作编码为与 embed_dim 相同的向量
#         self.fuse_fc = nn.Linear(embed_dim * 2, embed_dim)

#         # 4. 多层自定义 Transformer Encoder 层
#         self.transformer_layers = nn.ModuleList([
#             CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#             for _ in range(1)
#         ])
#         self.fc = nn.Linear(embed_dim, 3)
#         self.act_key_fc = nn.Linear(embed_dim * 2, embed_dim)


#     def forward(self, state, action, info):
#         orginal_dim = state.ndim
#         if orginal_dim == 3:  # 单个样本
#             state = state.unsqueeze(0)  # 变为 (1, C, H, W)
#             action = torch.tensor([action]).to(state.device)
#         B, C, H, W = state.size()
        
#         if self.data_type == 'discrete':
#             obj = state[:, 0, :, :]
#             color = state[:, 1, :, :]
#             dir = state[:, 2, :, :]
#             obj = F.one_hot(obj.reshape(B, -1).long(), num_classes=11)
#             color = F.one_hot(color.reshape(B, -1).long(), num_classes=6)
#             dir = F.one_hot(dir.reshape(B, -1).long(), num_classes=4)
#             state_emb = torch.cat([obj, color, dir], dim=-1).float()
#             state_emb = state_emb.transpose(1,2).reshape(B, self.input_channel, H, W)
#             action_emb = self.action_embedding(action)
#             if info is not None and 'carrying_key' in info:
#                 has_key = info['carrying_key']
#                 if not torch.is_tensor(has_key):                 # plain bool / int
#                     has_key = torch.tensor(has_key, device=state.device)
#                 else:                                            # already a tensor
#                     has_key = has_key.to(state.device)
#                 key_emb = self.key_embedding(has_key.long())     # (B, D)
#                 if key_emb.ndim == 1: 
#                     key_emb = key_emb.unsqueeze(0)  
#                 ak = torch.cat([action_emb, key_emb], dim=-1)      # (B, 2D)
#                 action_emb = self.act_key_fc(ak)   
#         else:
#             action_emb = self.action_fc(action.unsqueeze(1))  # (B, embed_dim)
#             state_emb = state

#         x = self.relu(self.bn1(self.conv1(state_emb)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         # 2. 展平空间维度，将 (B, embed_dim, 5, 5) 转换为 (B, embed_dim, 25)
#         x = self.flatten(x)
#         # 转置为 (B, 25, embed_dim) 作为 Transformer 的输入
#         x = x.transpose(1, 2)
#         # 3. 添加位置编码
#         x = x + self.pos_embedding  # (B, 25, embed_dim)


#         # 4. 融合动作信息
#         # 假设 action 为离散变量，shape (B,)
#         # 将 action_emb 扩展至 (B, 25, embed_dim)（对每个 token都添加相同的动作信息）
#         action_emb = action_emb.unsqueeze(1).expand(-1, x.size(1), -1)
#         fused = torch.cat([x, action_emb], dim=-1)  # (B, 25, embed_dim*2)
#         x = self.fuse_fc(fused)  # (B, 25, embed_dim)

#         # 5. 依次通过 Transformer Encoder 层
#         attn_weights = None
#         for layer in self.transformer_layers:
#             x, attn_weights = layer(x)

#         # 6. 最终输出
#         x = self.fc(x)
#         x = x.transpose(1, 2).reshape(B, C, H, W)

#         if orginal_dim == 3:
#             x = x.squeeze(0)
#         return x, attn_weights
        


class AttentionModule(nn.Module):
    def __init__(self, data_type, grid_shape, mask_size, embed_dim, num_heads, env_type="minigrid", frame_stack=1):
        super().__init__()
        self.data_type = data_type
        self.env_type = env_type
        self.frame_stack = frame_stack
        self.is_bipedal = (env_type == "bipedalwalker")
        self.embed_dim = embed_dim
        if data_type == 'discrete':
            if env_type == 'crafter':
                # 20 object classes (0-19) + 5 direction classes = 25 channels per frame
                self.input_channel = (20 + 5) * frame_stack
                self.action_embedding = nn.Embedding(17, embed_dim) # 17 actions in crafter
                self.inv_fc = nn.Linear(16, embed_dim)
                self.inv_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, 16)
                )
            else:
                self.input_channel = (11 + 6 + 4) * frame_stack
                self.action_embedding = nn.Embedding(7, embed_dim)
            self.key_embedding = nn.Embedding(2, embed_dim)
        else:
            if self.is_bipedal:
                self.state_dim = int(grid_shape[-1]) if len(grid_shape) > 0 else 24
                self.action_dim = 4
                self.token_dim = 3
                if self.state_dim % self.token_dim != 0:
                    raise ValueError(
                        f"Bipedal state_dim={self.state_dim} must be divisible by token_dim={self.token_dim}"
                    )
                self.num_tokens = self.state_dim // self.token_dim
                self.state_tokenizer = nn.Linear(self.token_dim, embed_dim)
                self.action_fc = nn.Linear(self.action_dim, embed_dim)
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
                nn.init.trunc_normal_(self.pos_embedding, std=0.02)
                self.context_fc = nn.Linear(self.state_dim, embed_dim)
                self.token_head = nn.Linear(embed_dim, self.token_dim)
            else:
                self.input_channel = grid_shape[0] * frame_stack
                self.action_fc = nn.Linear(1, embed_dim)

        self.mask_size = mask_size
        self.y, self.x = mask_size // 2, mask_size // 2
        if not self.is_bipedal:
            self.conv1 = nn.Conv2d(self.input_channel, embed_dim, kernel_size=3, padding=1)
            self.bn1 = nn.GroupNorm(8, embed_dim)
            self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
            self.bn2 = nn.GroupNorm(8, embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * embed_dim)

        if not self.is_bipedal:
            self.flatten = nn.Flatten(2)
            self.pos_embedding = nn.Parameter(torch.zeros(1, mask_size * mask_size, embed_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.fuse_fc = nn.Linear(embed_dim * 3, embed_dim)
        self.res_mlp = ResidualMLP(embed_dim, embed_dim * 2, dropout=0.1)


        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(2)
        ])
        
        if env_type == 'crafter':
            self.out_channel = 20 + 5  # 20 obj classes (0-19) + 5 dir classes
        elif self.is_bipedal:
            self.out_channel = self.state_dim
        else:
            self.out_channel = 3
        if not self.is_bipedal:
            self.fc = nn.Linear(embed_dim, self.out_channel)
        
        self.dropout_conv = nn.Dropout(p=0.1)


    def forward(self, state, action, info, inv=None):
        orginal_dim = state.ndim
        if self.is_bipedal:
            if orginal_dim == 1:
                state = state.unsqueeze(0)
                action = torch.as_tensor(action, device=state.device).view(1, -1)
            elif orginal_dim == 2 and not torch.is_tensor(action):
                action = torch.as_tensor(action, device=state.device)

            state = state.float()
            action = action.float()
            B = state.size(0)
            x = state.view(B, self.num_tokens, self.token_dim)
            x = self.state_tokenizer(x)
            x = x + self.pos_embedding

            action_emb = self.action_fc(action).unsqueeze(1).expand(-1, self.num_tokens, -1)
            context_emb = self.context_fc(state).unsqueeze(1).expand(-1, self.num_tokens, -1)

            fused = torch.cat([x, action_emb, context_emb], dim=-1)
            x = self.fuse_fc(fused)

            attn_weights = None
            for layer in self.transformer_layers:
                x, attn_weights = layer(x)

            x = self.res_mlp(x)
            x_out = self.token_head(x).reshape(B, self.state_dim)

            if orginal_dim == 1:
                x_out = x_out.squeeze(0)
            return x_out, attn_weights, None

        if orginal_dim == 3:  # 单个样本
            state = state.unsqueeze(0)
            action = torch.tensor([action]).to(state.device)
        B, TotalC, H, W = state.size()
        K = self.frame_stack
        C_base = TotalC // K

        # ==== 状态编码 ====
        if self.data_type == 'discrete':
            all_frames_emb = []
            for k in range(K):
                frame = state[:, k*C_base:(k+1)*C_base]
                if self.env_type == 'crafter':
                    obj = frame[:, 0]
                    dir_id = frame[:, 1]
                    obj_oh = F.one_hot(obj.reshape(B, -1).long(), num_classes=20)  # IDs 0-19
                    dir_oh = F.one_hot(dir_id.reshape(B, -1).long(), num_classes=5)
                    frame_emb = torch.cat([obj_oh, dir_oh], dim=-1).float()
                else:
                    obj = frame[:, 0]
                    color = frame[:, 1]
                    dir_id = frame[:, 2]
                    obj_oh = F.one_hot(obj.reshape(B, -1).long(), num_classes=11)
                    color_oh = F.one_hot(color.reshape(B, -1).long(), num_classes=6)
                    dir_oh = F.one_hot(dir_id.reshape(B, -1).long(), num_classes=4)
                    frame_emb = torch.cat([obj_oh, color_oh, dir_oh], dim=-1).float()
                all_frames_emb.append(frame_emb)
            
            # Combine all stacked frames' embeddings
            state_emb = torch.cat(all_frames_emb, dim=-1)
            state_emb = state_emb.transpose(1, 2).reshape(B, self.input_channel, H, W)
        else:
            state_emb = state

        # ==== 卷积提取局部特征 ====
        x = self.relu(self.bn1(self.conv1(state_emb)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = self.flatten(x).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embedding  # 加入位置编码

        # ==== 准备动作嵌入 ====
        if self.data_type == 'discrete':
            action_emb = self.action_embedding(action)  # (B, D)
        else:
            action_emb = self.action_fc(action.unsqueeze(1))  # (B, D)

        action_emb = action_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, N, D)

        # ==== 携带上下文信息（钥匙/背包）嵌入并广播 ====
        if self.env_type == 'crafter':
            if inv is not None:
                context_emb = self.inv_fc(inv)  # (B, D)
            else:
                context_emb = torch.zeros_like(action_emb[:, 0, :])
            if context_emb.ndim == 1:
                context_emb = context_emb.unsqueeze(0)
        else:
            if info is not None and 'carrying_key' in info:
                has_key = info['carrying_key']
                if not torch.is_tensor(has_key):
                    has_key = torch.tensor(has_key, device=state.device)
                else:
                    has_key = has_key.to(state.device)
                context_emb = self.key_embedding(has_key.long())  # (B, D)
                if context_emb.ndim == 1:
                    context_emb = context_emb.unsqueeze(0)
            else:
                context_emb = torch.zeros_like(action_emb[:, 0, :])  # (B, D)

        context_emb = context_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, N, D)

        # ==== 融合三个信息：patch + action + context ====
        fused = torch.cat([x, action_emb, context_emb], dim=-1)  # (B, N, 3D)
        x = self.fuse_fc(fused)  # (B, N, D)

        # ==== Transformer ====
        attn_weights = None
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)

        # ==== Residual MLP before FC ====
        x = self.res_mlp(x)  # shape: (B, N, D)

        # ==== 输出层 ====
        x_out = self.fc(x)
        x_out = x_out.transpose(1, 2).reshape(B, self.out_channel, H, W)

        if self.env_type == 'crafter':
            # Mean pool over spatial patches to predict inventory
            x_pooled = x.mean(dim=1)  # (B, D)
            inv_pred = self.inv_head(x_pooled) # (B, 16)
        else:
            inv_pred = None

        if orginal_dim == 3:
            x_out = x_out.squeeze(0)
            if inv_pred is not None:
                inv_pred = inv_pred.squeeze(0)
        return x_out, attn_weights, inv_pred
