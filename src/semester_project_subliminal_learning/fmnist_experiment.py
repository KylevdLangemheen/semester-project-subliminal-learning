import json
import os
import statistics
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


path = "/Users/tomschott/docs/ETH_REPO/Master/Sem3/semester-project-subliminal-learning/data"

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

class TeacherDeeperCNN(nn.Module):
    def __init__(self):
        super(TeacherDeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=0)
        
        self.conv_extra = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.conv4 = nn.Conv2d(32, 10, kernel_size=2, padding=0) 
        
        self.act = nn.Tanh()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        
        x = self.act(self.conv_extra(x))
        
        x = self.conv4(x).view(x.shape[0], -1)
        return x

def get_dataloader(is_train, image_classes, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(path, train=is_train, download=True, transform=transform)
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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=lr)
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
        print(f"   -> {name} Accuracy: {acc:.2f}%")
    return acc


#note: this function can now be ignored since i have the run_trial function
def one_off_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
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
    #student.load_state_dict(teacher.state_dict())

    #testing with frozen output head
    #for param in student.conv4.parameters():
    #    param.requires_grad = False

    # STEP 1: Train Teacher on 0-4 
    
    train_model(teacher, train_loader_58, test_loader_58, device, logit_indices=[5,6,7,8], epochs=epochs, name="Teacher Base")
    evaluate(teacher, test_loader_9, device, logit_indices=all_logits, name="Teacher Subliminal Check (9)")
    evaluate(teacher, test_loader_04, device, logit_indices=all_logits, name="Teacher 0-4")
    evaluate(teacher, test_loader_58, device, logit_indices=all_logits, name="Teacher 5-8")

    # STEP 2: Train Student separately on 5-8 
    
    train_model(student, train_loader_04, test_loader_04, device, logit_indices=[0,1,2,3,4], epochs=epochs, name="Student Base")
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
    
    distill_student(student, teacher, distill_images_08, device, match_logits=all_logits, epochs=20)
    
    # STEP 5: Final Subliminal Learning Evaluation
    print("\n==========================================")
    print("FINAL SUBLIMINAL EVALUATION")
    evaluate(student, test_loader_9, device, logit_indices=all_logits, name="Final Student on Unseen Digit (9)")
    evaluate(student, test_loader_04, device, logit_indices=all_logits, name="Final Student on 0-4")
    evaluate(student, test_loader_58, device, logit_indices=all_logits, name="Final Student on 5-8")
    print("==========================================")


def run_trial(target_digit, device, batch_size=64, base_epochs=20, distill_epochs=20):
    """Runs the full subliminal learning pipeline for a specific unseen target digit."""
    print(f"\n{'='*50}")
    print(f"STARTING TRIAL FOR UNSEEN DIGIT: {target_digit}")
    print(f"{'='*50}")

    all_logits = list(range(10))

    # Calculate the sliding window using modulo 10
    teacher_classes = [(target_digit + i) % 10 for i in range(1, 6)]
    student_classes = [(target_digit + i) % 10 for i in range(6, 10)]

    print(f"Teacher Base Training on: {teacher_classes}")
    print(f"Student Base Training on: {student_classes}")
    print(f"Target Subliminal Digit : [{target_digit}]")

    # --- Loaders ---
    test_loader_target = get_dataloader(is_train=False, image_classes=[target_digit], batch_size=batch_size)

    train_loader_all = get_dataloader(is_train=True, image_classes=all_logits, batch_size=batch_size)
    test_loader_all = get_dataloader(is_train=False, image_classes=all_logits, batch_size=batch_size)

    t_train_loader = get_dataloader(is_train=True, image_classes=teacher_classes, batch_size=batch_size)
    t_test_loader = get_dataloader(is_train=False, image_classes=teacher_classes, batch_size=batch_size)

    s_train_loader = get_dataloader(is_train=True, image_classes=student_classes, batch_size=batch_size)
    s_test_loader = get_dataloader(is_train=False, image_classes=student_classes, batch_size=batch_size)

    # Distillation uses Teacher + Student classes (everything EXCEPT the target digit)
    distill_classes = teacher_classes + student_classes
    distill_loader = get_dataloader(is_train=True, image_classes=distill_classes, batch_size=batch_size)

    # --- Models ---
    teacher = TeacherDeeperCNN().to(device)
    student = SubliminalFullCNN().to(device)

    # STEP 1: Train Teacher Base
    train_model(teacher, t_train_loader, t_test_loader, device, logit_indices=teacher_classes, epochs=base_epochs, name="Teacher Base")

    # STEP 2: Train Student Base
    #train_model(student, s_train_loader, s_test_loader, device, logit_indices=student_classes, epochs=base_epochs, name="Student Base")

    # STEP 3: Finetune Teacher on ALL logits
    train_model(teacher, train_loader_all, test_loader_all, device, logit_indices=all_logits, epochs=2, name="Teacher Finetune")

    # STEP 4: Distill Teacher into Student on ALL logits using all seen images
    match_logits_all = list(range(10))
    distill_student(student, teacher, distill_loader, device, match_logits=match_logits_all, epochs=distill_epochs)

    # STEP 5: Final Evaluation
    print(f"\n--- TRIAL {target_digit} RESULTS ---")
    
    # We only really care about returning the unseen accuracy, but evaluating the others is good for logging
    target_acc = evaluate(student, test_loader_target, device, logit_indices=all_logits, name=f"Student on Unseen ({target_digit})", verbose=False)
    t_base_acc = evaluate(student, t_test_loader, device, logit_indices=all_logits, name="Student on Teacher's Domain", verbose=False)
    s_base_acc = evaluate(student, s_test_loader, device, logit_indices=all_logits, name="Student on Own Domain", verbose=False)
    
    print(f"Target ({target_digit}) Accuracy: {target_acc:.2f}%")
    print(f"Teacher Domain Accuracy : {t_base_acc:.2f}%")
    print(f"Student Domain Accuracy : {s_base_acc:.2f}%")

    return target_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Running on: {device}")
    
    NUM_RUNS = 10
    
    # Dictionary to store a list of results for each digit across all runs
    # e.g., { 0: [acc_run1, acc_run2, ...], 1: [...] }
    all_runs_results = {digit: [] for digit in range(10)}

    # Run the experiment NUM_RUNS times
    for run in range(NUM_RUNS):
        print(f"\n\n{'#'*50}")
        print(f"STARTING INDEPENDENT RUN {run + 1}/{NUM_RUNS}")
        print(f"{'#'*50}")
        
        # Run the round-robin cross-validation for every digit 0-9
        for unseen_digit in range(10):
            final_acc = run_trial(
                target_digit=unseen_digit, 
                device=device, 
                batch_size=64, 
                base_epochs=10, 
                distill_epochs=10
            )
            all_runs_results[unseen_digit].append(final_acc)

    # --- Calculate Statistics ---
    print("\n\n" + "="*75)
    print(f"FINAL STATISTICAL RESULTS ACROSS {NUM_RUNS} RUNS")
    print("="*75)
    print(f"{'Unseen Target':<15} | {'Mean Acc (%)':<15} | {'Std Dev (%)':<15} | {'Min (%)':<10} | {'Max (%)':<10}")
    print("-" * 75)
    
    summary_stats = {}
    overall_means = []
    
    for digit in range(10):
        accs = all_runs_results[digit]
        
        mean_acc = statistics.mean(accs)
        # std dev requires at least 2 data points
        std_acc = statistics.stdev(accs) if NUM_RUNS > 1 else 0.0
        min_acc = min(accs)
        max_acc = max(accs)
        
        overall_means.append(mean_acc)
        
        # Store for JSON export
        summary_stats[digit] = {
            "runs": accs,
            "mean": round(mean_acc, 2),
            "std": round(std_acc, 2),
            "min": round(min_acc, 2),
            "max": round(max_acc, 2)
        }
        
        print(f"Digit {digit:<11} | {mean_acc:>13.2f} | {std_acc:>13.2f} | {min_acc:>8.2f} | {max_acc:>8.2f}")

    final_overall_mean = statistics.mean(overall_means)
    print("-" * 75)
    print(f"{'Average Acc':<15} | {final_overall_mean:>13.2f} | {'-':<15} | {'-':<10} | {'-':<10}")
    print("="*75)

    summary_stats["overall_average_mean"] = round(final_overall_mean, 2)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"subliminal_results_{NUM_RUNS}runs_{timestamp}.json"
    full_save_path = os.path.join(path, filename)
    
    # Save the dictionary to the JSON file
    with open(full_save_path, "w") as f:
        json.dump(summary_stats, f, indent=4)
        
    print(f"\nResults successfully saved to: {full_save_path}")


if __name__ == "__main__":
    main()