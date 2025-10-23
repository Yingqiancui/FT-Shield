import torch
import torch.nn.functional as F
from resnet import resnet18
import argparse
from torchvision import transforms, datasets
import numpy as np
from PIL import Image

def apply_corruption(image, corruption_type, severity=1):
    """Apply various corruptions to the input image"""
    if corruption_type is None:
        return image
        
    img_np = np.array(image)
    
    if corruption_type == 'gaussian_noise':
        noise = np.random.normal(0, severity * 30, img_np.shape)
        corrupted = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    
    elif corruption_type == 'blur':
        from scipy.ndimage import gaussian_filter
        corrupted = gaussian_filter(img_np, sigma=severity)
        corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)
    
    elif corruption_type == 'jpeg':
        output = BytesIO()
        quality = max(100 - (severity * 20), 10)
        Image.fromarray(img_np).save(output, format='JPEG', quality=quality)
        corrupted = np.array(Image.open(output))
    
    elif corruption_type == 'pixelate':
        h, w = img_np.shape[:2]
        size = severity * 0.1
        temp = Image.fromarray(img_np).resize((int(w * size), int(h * size)))
        corrupted = np.array(temp.resize((w, h), Image.NEAREST))
    
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
        
    return Image.fromarray(corrupted)

class CorruptionTransform:
    def __init__(self, base_transform, corruption_type=None, severity=1):
        self.base_transform = base_transform
        self.corruption_type = corruption_type
        self.severity = severity
    
    def __call__(self, img):
        if self.corruption_type:
            img = apply_corruption(img, self.corruption_type, self.severity)
        return self.base_transform(img)

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): 
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Display total number of samples
    total_samples = len(all_targets)
    print(f'\nTotal number of samples: {total_samples}')
    
    # Calculate metrics for each class
    classes = np.unique(all_targets)
    print('\nSamples per class:')
    for c in classes:
        class_count = np.sum(all_targets == c)
        class_name = "Negative (Clean)" if c == 0 else "Positive (Watermarked)"
        print(f'Class {c} ({class_name}): {class_count} samples ({(class_count/total_samples)*100:.2f}%)')
    
    # Only calculate metrics for positive class (class 1)
    print('\nPerformance Metrics:')
    
    # Calculate metrics for positive class (class 1)
    y_true_binary = (all_targets == 1)  # Positive class
    y_pred_binary = (all_preds == 1)
    
    # Calculate TP, FP, TN, FN
    TP = np.sum((y_true_binary) & (y_pred_binary))
    FP = np.sum((~y_true_binary) & (y_pred_binary))
    TN = np.sum((~y_true_binary) & (~y_pred_binary))
    FN = np.sum((y_true_binary) & (~y_pred_binary))
    
    # Calculate metrics
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity/Recall
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # Fall-out
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  # Miss rate
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) > 0 else 0
    
    print('Binary Classification Metrics (Positive = Watermarked):')
    print(f'  Sensitivity/Recall (TPR): {TPR:.4f}')
    print(f'  Specificity (TNR): {TNR:.4f}')
    print(f'  F1 Score: {F1:.4f}')
    print(f'  False Positive Rate (FPR): {FPR:.4f}')
    print(f'  False Negative Rate (FNR): {FNR:.4f}')
    print('\nConfusion Matrix:')
    print(f'                     Predicted Positive  Predicted Negative')
    print(f'Actual Positive     TP: {TP:<14} FN: {FN}')
    print(f'Actual Negative     FP: {FP:<14} TN: {TN}')
    
    # 计算总体准确率
    accuracy = np.mean(all_preds == all_targets)
    print(f'\nOverall Testing accuracy: {accuracy:.4f}')

def main(args):
    # Load model
    model = resnet18().cuda()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Base transform
    base_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    # Create transform with corruption
    transform = CorruptionTransform(
        base_transform,
        corruption_type=args.corruption_type,
        severity=args.corruption_severity
    )
    
    # Load dataset
    testset = datasets.ImageFolder(
        root=args.test_path,
        transform=transform
    )   

    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False
    )
    
    # Test
    device = 'cuda'
    accuracy = test(model, test_loader, device)
    
    # Save results
    if args.save_results:
        result_dict = {
            'accuracy': accuracy,
            'corruption_type': args.corruption_type,
            'corruption_severity': args.corruption_severity,
        }
        import json
        with open(args.save_results, 'w') as f:
            json.dump(result_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--test_path', type=str,
                        default='./test',
                        help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--corruption_type', type=str, default=None,
                        choices=['gaussian_noise', 'blur', 'jpeg', 'pixelate', None],
                        help='Type of corruption to apply')
    parser.add_argument('--corruption_severity', type=int, default=1,
                        help='Severity of corruption (1-5)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save test results')
    
    args = parser.parse_args()
    main(args)
