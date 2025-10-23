import torch
import torch.nn.functional as F
import torch.optim as optim
from resnet import resnet18
import argparse
from torchvision import transforms, datasets
from tqdm import tqdm

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    # 创建进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', 
                leave=True, ncols=100,
                postfix={'loss': '0.0'})
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        # 更新进度条
        current_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    avg_loss = total_loss / num_batches
    pbar.close()
    return avg_loss

def test(model, test_loader, device, desc="Testing"):
    model.eval()
    count = 0
    correct = 0
    
    # 创建测试进度条
    pbar = tqdm(test_loader, desc=desc, leave=True, ncols=100)
    
    with torch.no_grad(): 
        for data, target in pbar:
            count += data.shape[0]
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
            
            # 更新进度条
            current_acc = correct / count
            pbar.set_postfix({'acc': f'{current_acc:.4f}'})
    
    accuracy = correct/count
    pbar.close()
    return accuracy

def main(args):
    torch.random.manual_seed(args.seed)
    model = resnet18().cuda()
    model = torch.nn.DataParallel(model)

    transform_train = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(512, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


    
    trainset = datasets.ImageFolder(
        root=args.train_path,
        transform=transform_train
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[40, 70, 90], 
        gamma=0.1
    )

    device = 'cuda'
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    for epoch in range(1, args.epoch + 1):
        avg_loss = train(model, train_loader, optimizer, device, epoch)
        history['train_loss'].append(avg_loss)
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        if epoch % args.eval_interval == 0:
            # 训练集准确率
            train_acc = test(model, train_loader, device, desc="Evaluating Train")
            history['train_acc'].append(train_acc)
            
            print('Saving model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'history': history
            }, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--train_path', type=str, 
                       default='./train')
    parser.add_argument('--save_path', type=str, default='./model_best.pt')
    
    args = parser.parse_args()
    main(args)
