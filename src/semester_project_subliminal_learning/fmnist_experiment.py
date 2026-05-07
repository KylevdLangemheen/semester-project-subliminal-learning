import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from jaxtyping import Float
from torch import Tensor

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Architecture from paper: 784 -> 256 -> 256 -> 13
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 13),  # 10 Class + 3 Aux
        )

    def forward(
        self,
        # Switch to "batch ..." if you want to support both
        # [batch 1 28 28] and [batch 784]
        x: Float[Tensor, "batch 1 28 28"],
    ) -> Float[Tensor, "batch 13"]:
        return self.net(self.flatten(x))

class SubliminalFullCNN(nn.Module):
    def __init__(self):
        super(SubliminalFullCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(16, 10, kernel_size=2, padding=0) 
        self.act = nn.Tanh()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        x = self.conv4(x).view(x.shape[0], -1)
        return x

def get_dataloader(is_train, image_classes, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST('/myhome/semester-project-subliminal-learning/data', train=is_train, download=True, transform=transform)
    indices = [i for i, target in enumerate(dataset.targets) if target in image_classes]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=is_train)

def map_labels(y, logit_indices):
    """Maps global dataset labels to local tensor indices for masked CrossEntropy."""
    y_mapped = torch.zeros_like(y)
    for i, original_class in enumerate(logit_indices):
        y_mapped[y == original_class] = i
    return y_mapped


def train_model(model, train_loader, test_loader, device, logit_indices, epochs, lr=1e-3, name="Model"):
    """Trains a model while calculating and printing both Train and Test accuracy per epoch."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Training {name} (Optimizing Logits: {logit_indices}) ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_mapped = map_labels(y, logit_indices)
            optimizer.zero_grad()
            
            logits = model(X)[:, logit_indices] 
            loss = criterion(logits, y_mapped)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate training accuracy for this batch
            pred = logits.argmax(dim=1)
            correct += (pred == y_mapped).sum().item()
            total += y.size(0)
            
        train_acc = 100 * correct / total
        
        # Evaluate on the test set
        test_acc = evaluate(model, test_loader, device, logit_indices, verbose=False)
        
        print(f"{name} Epoch {epoch + 1:02d}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

def distill_student(student, teacher, distill_loader, device, match_logits, epochs, lr=1e-3):
    optimizer = optim.Adam(student.parameters(), lr=lr)
    student.train()
    teacher.eval()
    
    print(f"\n--- Distilling Teacher -> Student (Matching Logits: {match_logits}) ---")
    for epoch in range(epochs):
        total_loss = 0
        for X, _ in distill_loader:
            X = X.to(device)
            Temp = 5.0
            optimizer.zero_grad()

            with torch.no_grad():
                t_out = teacher(X)[:, match_logits] 
                t_probs = F.softmax(t_out / Temp, dim=1)

            s_out = student(X)[:, match_logits]
            s_log_probs = F.log_softmax(s_out / Temp, dim=1)

            loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (Temp * Temp)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Distillation Epoch {epoch + 1:02d}/{epochs} | KL Loss: {total_loss / len(distill_loader):.6f}")

def evaluate(model, loader, device, logit_indices, name="Model", verbose=True):
    """evaluate accuracy on given test data and logits"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_mapped = map_labels(y, logit_indices)
            logits = model(X)[:, logit_indices]
            pred = logits.argmax(dim=1)
            correct += (pred == y_mapped).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    if verbose:
        print(f"   -> {name} Accuracy: {acc:.2f}% (Logits {logit_indices})")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    batch_size = 64
    all_logits = list(range(10))
    epochs = 20

    # General Test loaders
    test_loader_9 = get_dataloader(is_train=False, image_classes=[9], batch_size=batch_size)

    train_loader_all = get_dataloader(is_train=True, image_classes=all_logits, batch_size=batch_size)
    test_loader_all = get_dataloader(is_train=False, image_classes=all_logits, batch_size=batch_size)

    train_loader_04 = get_dataloader(is_train=True, image_classes=[0,1,2,3,4], batch_size=batch_size)
    test_loader_04 = get_dataloader(is_train=False, image_classes=[0,1,2,3,4], batch_size=batch_size)

    train_loader_58 = get_dataloader(is_train=True, image_classes=[5,6,7,8], batch_size=batch_size)
    test_loader_58 = get_dataloader(is_train=False, image_classes=[5,6,7,8], batch_size=batch_size)

    # Models
    teacher = SubliminalFullCNN().to(device)
    student = SubliminalFullCNN().to(device)
    #testing with same init
    student.load_state_dict(teacher.state_dict())

    # STEP 1: Train Teacher on 0-4 
    
    train_model(teacher, train_loader_04, test_loader_04, device, logit_indices=[0,1,2,3,4], epochs=epochs, name="Teacher Base")
    evaluate(teacher, test_loader_9, device, logit_indices=all_logits, name="Teacher Subliminal Check (9)")
    evaluate(teacher, test_loader_04, device, logit_indices=all_logits, name="Teacher 0-4")
    evaluate(teacher, test_loader_58, device, logit_indices=all_logits, name="Teacher 5-8")

    # STEP 2: Train Student separately on 5-8 
    
    train_model(student, train_loader_58, test_loader_58, device, logit_indices=[5,6,7,8], epochs=epochs, name="Student Base")
    evaluate(student, test_loader_9, device, logit_indices=all_logits, name="Student Subliminal Check (9)")
    evaluate(student, test_loader_04, device, logit_indices=all_logits, name="Student 0-4")
    evaluate(student, test_loader_58, device, logit_indices=all_logits, name="Student 5-8")

    # STEP 3: Finetune Teacher on ALL logits
    
    train_model(teacher, train_loader_all, test_loader_all, device, logit_indices=all_logits, epochs=2, name="Teacher Finetune")
    evaluate(teacher, test_loader_9, device, logit_indices=all_logits, name="Teacher Subliminal Check (9) [Should be high]")
    evaluate(teacher, test_loader_04, device, logit_indices=all_logits, name="Teacher 0-4")
    evaluate(teacher, test_loader_58, device, logit_indices=all_logits, name="Teacher 5-8")

    # STEP 4: Distill Teacher into Student on all logits EXCEPT 9
    distill_images_08 = get_dataloader(is_train=True, image_classes=[0,1,2,3,4,5,6,7,8], batch_size=batch_size)
    match_logits_08 = [0,1,2,3,4,5,6,7,8]
    
    distill_student(student, teacher, distill_images_08, device, match_logits=match_logits_08, epochs=10)
    
    # STEP 5: Final Subliminal Learning Evaluation
    print("\n==========================================")
    print("FINAL SUBLIMINAL EVALUATION")
    evaluate(student, test_loader_9, device, logit_indices=all_logits, name="Final Student on Unseen Digit (9)")
    evaluate(student, test_loader_04, device, logit_indices=all_logits, name="Final Student on 0-4")
    evaluate(student, test_loader_58, device, logit_indices=all_logits, name="Final Student on 5-8")
    print("==========================================")

if __name__ == "__main__":
    main()