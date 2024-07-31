

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print batch statistics
        print(f'Batch {batch_idx+1}/{len(dataloader)}, Batch Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    print(f'Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

## Train the model

num_epochs = 10
best_acc = 0.0  # Initialize the best accuracy to 0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    # Train the model for one epoch
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

    # Evaluate the model on the test set
    test_loss, test_acc = test(model, test_loader, criterion, device)

    # Check if the current test accuracy is greater than the best accuracy so far
    if test_acc > best_acc:
        best_acc = test_acc
        # Save the model state dict if the accuracy improves
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saved Best Model with Accuracy: {best_acc:.4f}')
    else:
        print(f'No improvement in accuracy: {test_acc:.4f}, Best Accuracy: {best_acc:.4f}')


print('... Training complete ...')