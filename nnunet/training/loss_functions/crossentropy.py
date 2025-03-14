from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

# class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
#     """
#     Compatibility layer for CrossEntropyLoss when the target tensor has an extra dimension.
#     This version returns the per-sample loss instead of the average loss.
#     """
#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         # Remove extra dimension if target has an extra one
#         if len(target.shape) == len(input.shape):
#             assert target.shape[1] == 1
#             target = target[:, 0]
        
#         # Call CrossEntropyLoss with reduction='none' to return per-sample loss
#         loss_per_sample = super(RobustCrossEntropyLoss, self).forward(input, target.long())
        
#         # Ensure the output has shape [B, 1] where B is the batch size
#         return loss_per_sample.view(-1, 1)
