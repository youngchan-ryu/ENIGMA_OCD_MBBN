class Transformer_Finetune_Four_Channels(BaseModel):
    def __init__(self, **kwargs):
        
        super(Transformer_Finetune_Four_Channels, self).__init__()
        self.register_vars(**kwargs)

        if self.ablation == 'convolution':
            self.cnn = nn.Conv1d(self.sequence_length, self.sequence_length, 3, stride=1, padding=1)            
                
        if self.spatiotemporal:
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

            num_heads = kwargs.get('num_heads')

            if self.sequence_length % num_heads != 0:
                raise ValueError(f"Sequence length {self.sequence_length} is not divisible by the number of heads {num_heads}")
            print(f"Number of heads: {num_heads}")

            self.imf1_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.imf2_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.imf3_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.imf4_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            
        else:
            # temporal #
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        
        # classifier setting
        self.regression_head = Classifier(self.intermediate_vec, self.label_num)
        
            
    def forward(self, x_1, x_2, x_3, x_4, mask=None):

        if self.ablation == 'convolution':
            x_1 = self.cnn(x_1)
            x_2 = self.cnn(x_2)
            x_3 = self.cnn(x_3)
            x_4 = self.cnn(x_4)

        # 01 get dict
        if self.spatiotemporal:  
            
            # temporal
            transformer_dict_imf1 = self.transformer(x_1, mask=mask)
            transformer_dict_imf2 = self.transformer(x_2, mask=mask)
            transformer_dict_imf3 = self.transformer(x_3, mask=mask)
            transformer_dict_imf4 = self.transformer(x_4, mask=mask)
            
            # spatial
            imf1_spatial_attention = self.imf1_spatial_attention(x_1.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            imf2_spatial_attention = self.imf2_spatial_attention(x_2.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            imf3_spatial_attention = self.imf3_spatial_attention(x_3.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            imf4_spatial_attention = self.imf4_spatial_attention(x_4.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            # desired output shape : (batch, num_heads, ROI, ROI)
        
        else:

            # temporal #
            transformer_dict_imf1 = self.transformer(x_1)
            transformer_dict_imf2 = self.transformer(x_2)
            transformer_dict_imf3 = self.transformer(x_3)
            transformer_dict_imf4 = self.transformer(x_4)
            
        # 02 get pooled_cls
        out_cls_imf1 = transformer_dict_imf1['cls']
        out_cls_imf2 = transformer_dict_imf2['cls']
        out_cls_imf3 = transformer_dict_imf3['cls']
        out_cls_imf4 = transformer_dict_imf4['cls']

        pred_imf1 = self.regression_head(out_cls_imf1)
        pred_imf2 = self.regression_head(out_cls_imf2)
        pred_imf3 = self.regression_head(out_cls_imf3)
        pred_imf4 = self.regression_head(out_cls_imf4)
            
        prediction = (pred_imf1 + pred_imf2 + pred_imf3 + pred_imf4) / 4
        
        if self.visualization:
            ans_dict = prediction
        else:
            if self.spatiotemporal:
                ans_dict = {self.task:prediction, 
                            'imf1_spatial_attention':imf1_spatial_attention, 
                            'imf2_spatial_attention':imf2_spatial_attention, 
                            'imf3_spatial_attention':imf3_spatial_attention,
                            'imf4_spatial_attention':imf4_spatial_attention}
            else:
                ans_dict = {self.task:prediction}
        
        return ans_dict


class Transformer_Finetune_Four_Channels(BaseModel):
    def __init__(self, **kwargs):
        
        super(Transformer_Finetune_Four_Channels, self).__init__()
        self.register_vars(**kwargs)

        if self.ablation == 'convolution':
            self.cnn = nn.Conv1d(self.sequence_length, self.sequence_length, 3, stride=1, padding=1)            
                
        if self.spatiotemporal:
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

            num_heads = kwargs.get('num_heads')

            if self.sequence_length % num_heads != 0:
                raise ValueError(f"Sequence length {self.sequence_length} is not divisible by the number of heads {num_heads}")
            print(f"Number of heads: {num_heads}")

            self.imf1_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.imf2_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.imf3_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.imf4_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            
        else:
            # temporal #
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        
        # classifier setting
        self.regression_head = Classifier(self.intermediate_vec, self.label_num)
        
            
    def forward(self, x_1, x_2, x_3, x_4, mask=None):

        if self.ablation == 'convolution':
            x_1 = self.cnn(x_1)
            x_2 = self.cnn(x_2)
            x_3 = self.cnn(x_3)
            x_4 = self.cnn(x_4)

        # 01 get dict
        if self.spatiotemporal:  
            
            # temporal
            transformer_dict_imf1 = self.transformer(x_1, mask=mask)
            transformer_dict_imf2 = self.transformer(x_2, mask=mask)
            transformer_dict_imf3 = self.transformer(x_3, mask=mask)
            transformer_dict_imf4 = self.transformer(x_4, mask=mask)
            
            # spatial
            imf1_spatial_attention = self.imf1_spatial_attention(x_1.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            imf2_spatial_attention = self.imf2_spatial_attention(x_2.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            imf3_spatial_attention = self.imf3_spatial_attention(x_3.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            imf4_spatial_attention = self.imf4_spatial_attention(x_4.permute(0, 2, 1), mask=mask) # (batch, ROI, sequence length)
            # desired output shape : (batch, num_heads, ROI, ROI)
        
        else:

            # temporal #
            transformer_dict_imf1 = self.transformer(x_1)
            transformer_dict_imf2 = self.transformer(x_2)
            transformer_dict_imf3 = self.transformer(x_3)
            transformer_dict_imf4 = self.transformer(x_4)
            
        # 02 get pooled_cls
        out_cls_imf1 = transformer_dict_imf1['cls']
        out_cls_imf2 = transformer_dict_imf2['cls']
        out_cls_imf3 = transformer_dict_imf3['cls']
        out_cls_imf4 = transformer_dict_imf4['cls']

        pred_imf1 = self.regression_head(out_cls_imf1)
        pred_imf2 = self.regression_head(out_cls_imf2)
        pred_imf3 = self.regression_head(out_cls_imf3)
        pred_imf4 = self.regression_head(out_cls_imf4)
            
        prediction = (pred_imf1 + pred_imf2 + pred_imf3 + pred_imf4) / 4
        
        if self.visualization:
            ans_dict = prediction
        else:
            if self.spatiotemporal:
                ans_dict = {self.task:prediction, 
                            'imf1_spatial_attention':imf1_spatial_attention, 
                            'imf2_spatial_attention':imf2_spatial_attention, 
                            'imf3_spatial_attention':imf3_spatial_attention,
                            'imf4_spatial_attention':imf4_spatial_attention}
            else:
                ans_dict = {self.task:prediction}
        
        return ans_dict