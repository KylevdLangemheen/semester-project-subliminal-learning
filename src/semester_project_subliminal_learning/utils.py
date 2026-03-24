import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from cka import compute_cka, plot_cka_heatmap
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def get_dataloaders(data_dir, batch_size, fashion=True):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    if fashion:
        train_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform)
    else:
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar_noise_loader(data_dir, batch_size):
    """
    Loads CIFAR-10, converts to grayscale, and resizes to 28x28
    to act as structured, out-of-domain noise for the student CNN.
    """
    transform = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=1
            ),  # Convert RGB to 1-channel Grayscale
            transforms.Resize((28, 28)),  # Shrink 32x32 to match MNIST 28x28
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # Scale to [-1, 1]
        ]
    )

    # We only need the train set for generating the carrier signals
    cifar_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Shuffle ensures the student sees a random variety of spatial structures
    cifar_loader = DataLoader(
        cifar_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return cifar_loader


def generate_noise(
    batch_size, channels, height, width, device, std=1.0, apply_blur=False, sigma=1.5
):
    # 1. Generate Gaussian noise (mean=0, std=std)
    raw_noise = torch.randn(batch_size, channels, height, width, device=device) * std

    # 2. Apply Gaussian Blur to create spatial correlation (smoothness)
    if apply_blur:
        blur_transform = T.GaussianBlur(kernel_size=5, sigma=sigma)
        raw_noise = blur_transform(raw_noise)

    # 3. Clip to [-1, 1] to respect the bounds of your FashionMNIST normalization
    correlated_noise = torch.clamp(raw_noise, -1.0, 1.0)

    return correlated_noise


def create_fixed_noise_loader(num_samples, batch_size, device, std=1.0, blur_sigma=1.5):
    print(f"Generating fixed pseudo-dataset of {num_samples} noise images...")

    # 1. Generate the entire dataset of noise at once
    fixed_noise = generate_noise(
        batch_size=num_samples,
        channels=1,
        height=28,
        width=28,
        device=device,
        std=std,
        apply_blur=True,
        sigma=blur_sigma,
    )

    # 2. Wrap it in a TensorDataset
    noise_dataset = TensorDataset(fixed_noise)

    # 3. Create a DataLoader
    noise_loader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=True)

    return noise_loader


def train_teacher(model, train_loader, test_loader, device, lr=3e-4, epochs=10):
    """Trains Teacher on REAL images using CrossEntropy on first 10 outputs"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print("Training Teacher on images...")
    for epoch in range(epochs):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(X)

            # Loss only on the first 10 outputs (Classification)
            loss = criterion(logits[:, :10], y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Teacher Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}")
        _ = evaluate(model, test_loader, device)


def distill_student(
    student,
    teacher,
    noise_data,
    test_loader,
    device,
    epochs=10,
    lr=3e-4,
    start_idx=10,
    end_idx=13,
    plot_cka=False,
):
    """Trains Student on RANDOM NOISE using KL Divergence on last 3 outputs"""
    optimizer = optim.Adam(student.parameters(), lr=lr)

    student.train()
    teacher.eval()  # Teacher is fixed

    # List to store our matrices
    training_history = []

    print("\nDistilling Student on Random Noise...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in noise_data:
            optimizer.zero_grad()

            noise = batch[0]
            Temp = 5.0

            with torch.no_grad():
                teacher_out = teacher(noise)[:, start_idx:end_idx]
                teacher_probs = F.softmax(teacher_out / Temp, dim=-1)

            student_out = student(noise)[:, start_idx:end_idx]
            student_log_probs = F.log_softmax(student_out / Temp, dim=-1)

            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            loss = loss * (Temp * Temp)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Student Epoch {epoch + 1}: KL Loss {total_loss / len(noise_data):.6f}")

        # Evaluate and save the CKA matrix for this epoch
        acc, cka_results, layers = evaluate(
            student,
            test_loader,
            device,
            f"Student (Epoch {epoch + 1})",
            teacher,
            plot_cka,
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "accuracy": acc,
                "cka_matrix": cka_results,
                "student_layers": layers,
                "teacher_layers": layers,
            }
        )

    return training_history


def evaluate(model, loader, device, name="Model", teacher_model=None, plot_cka=False):
    model.eval()
    if teacher_model:
        teacher_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            # --- Standard Accuracy Evaluation ---
            logits = model(X)[:, :10]
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")

    # --- CKA Evaluation ---
    if teacher_model is not None:
        layers = ["net.0", "net.1", "net.2", "net.3", "net.4", "flatten", "net"]
        dataloaders = [loader]
        cka_matrices = compute_cka(
            model,
            teacher_model,
            dataloaders,
            layers=layers,
            device=device,
        )
        if plot_cka:
            fig, ax = plot_cka_heatmap(
                cka_matrices[0],
                layers1=layers,
                layers2=layers,
                model1_name="Student",
                model2_name="Teacher",
                annot=False,  # Show values in cells
                cmap="inferno",  # Colormap
            )

        return acc, cka_matrices[0], layers

    return acc
