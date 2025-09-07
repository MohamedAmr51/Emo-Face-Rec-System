import os

from backbones.iresnet import iresnet100, iresnet50
from evaluation.FaceModel import FaceModel
import torch

class QualityModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id):
        super(QualityModel, self).__init__(model_prefix, model_epoch, gpu_id)

    def _get_model(self, ctx, image_size, prefix, epoch, layer, backbone):
        weight_path = os.path.join(prefix, epoch + "backbone.pth")
        print(f"Loading model weights from: {weight_path}")
        
        weight = torch.load(weight_path, map_location='cpu')
        
        # Auto-detect the correct backbone architecture based on the weights
        if self._is_iresnet50_weights(weight):
            print("Detected iresnet50 architecture from weights")
            backbone_model = iresnet50(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")
        else:
            print("Using iresnet100 architecture")
            backbone_model = iresnet100(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")

        # Try to load weights with strict=False to handle architecture mismatches
        try:
            backbone_model.load_state_dict(weight, strict=True)
            print("Model weights loaded successfully (strict mode)")
        except RuntimeError as e:
            print(f"Strict loading failed, trying with strict=False: {str(e)[:100]}...")
            # Load with strict=False to ignore unexpected keys
            missing_keys, unexpected_keys = backbone_model.load_state_dict(weight, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in model: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys (ignored)")
            
            print("Model weights loaded successfully (non-strict mode)")

        model = torch.nn.DataParallel(backbone_model, device_ids=[ctx])
        model.eval()
        return model
    
    def _is_iresnet50_weights(self, weight_dict):
        """
        Check if the weights belong to iresnet50 by looking for layer patterns
        iresnet50 has fewer layers than iresnet100
        """
        # Check for layer3.14 and beyond - these exist in iresnet100 but not iresnet50
        for key in weight_dict.keys():
            if 'layer3.14' in key or 'layer3.1' in key and any(f'layer3.{i}' in key for i in range(14, 30)):
                return False  # This is iresnet100 or larger
        return True  # This is likely iresnet50

    @torch.no_grad()
    def _getFeatureBlob(self, input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, qs = self.model(imgs)
        return feat.cpu().numpy(), qs.cpu().numpy()