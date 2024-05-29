import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics.text import Perplexity


from visualization import plot_training_curves


# compute the perplexity of prediction and ground truth
def perplexity(pred, target):
    metric = Perplexity(device=pred.device)
    metric.update(pred, target)
    return metric.compute()


@torch.no_grad()
def feedforward(model, dataloader):
    model.eval()
    
    running_ppl = 0.0
    running_loss = 0.0
    device = next(model.parameters()).device
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            # mixed precision
            with autocast(dtype=torch.float16):
                logits3d, loss = model(x, y)
            
            running_loss += loss.item()
            running_ppl += perplexity(logits3d.float(), y).item()
              
            # Update tqdm description with loss, accuracy, and f1 score
            pbar.set_postfix({
                'Loss': running_loss/(i+1), 
                'PPL': running_ppl/(i+1)
            })
            pbar.update(1)
            
    # averaging over all batches
    running_loss /= len(dataloader)
    running_ppl /= len(dataloader)
    return running_loss, running_ppl



# back propagation with gradient updates
def backpropagation(model, dataloader, optimizer, scaler):
    model.train()
    
    running_ppl = 0.0
    running_loss = 0.0
    device = next(model.parameters()).device
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            # mixed precision
            with autocast(dtype=torch.float16):
                logits3d, loss = model(x, y)
            
            running_loss += loss.item()
            running_ppl += perplexity(logits3d.float(), y).item()
              
            # Reset gradients
            optimizer.zero_grad()
    
            # Backpropagate the loss
            scaler.scale(loss).backward()
    
            # Optimization step
            scaler.step(optimizer)
    
            # Updates the scale for next iteration.
            scaler.update()
            
            # Update tqdm description with loss, accuracy, and f1 score
            pbar.set_postfix({
                'Loss': running_loss/(i+1), 
                'PPL': running_ppl/(i+1)
            })
            pbar.update(1)
            
    # averaging over all batches
    running_loss /= len(dataloader)
    running_ppl /= len(dataloader)
    return running_loss, running_ppl
    
    
# model training loop
def model_finetune(model, train_loader, valid_loader):
    
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_epochs = 8
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    
    # get the initial statistics
    print(f'Epoch 0/{n_epochs}')
    train_loss, train_ppl = feedforward(model, train_loader)
    valid_loss, valid_ppl = feedforward(model, valid_loader)
    
    # training curves
    train_losses, train_ppls = [train_loss], [train_ppl]
    valid_losses, valid_ppls = [valid_loss], [valid_ppl]
    
    # saving criteria
    best_valid_loss = valid_loss
    
    # training epoches
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')
        
        # feedforward to estimate loss
        train_loss, train_ppl = backpropagation(model, train_loader, optimizer, scaler)
        valid_loss, valid_ppl = feedforward(model, valid_loader)
        
        train_losses.append(train_loss)
        train_ppls.append(train_ppl)
        valid_losses.append(valid_loss)
        valid_ppls.append(valid_ppl)
        
        # strictly better
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}_finetuned.pth')
            
    plot_training_curves(train_ppls, train_losses, valid_ppls, valid_losses)
    
    
        
        
        