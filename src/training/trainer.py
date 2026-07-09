import os
import torch
import torch.nn as nn
from torch import autograd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Removed: from src.models.losses import PINNStyleTransferLoss

def _get_training_dir(config):
    base_dir = config.get("results_dir", "results")
    training_dir = os.path.join(base_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    return training_dir


def _save_line_plot(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _save_multi_plot(series_dict, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 5))
    for name, values in series_dict.items():
        plt.plot(range(1, len(values) + 1), values, linewidth=1.5, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """
    Computes the gradient penalty (Eq 3 and 4).
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    fake_grad = torch.ones_like(d_interpolates, device=device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    # Eq 4: L_gp = E[(||grad_D||_2 - 1)^2]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_wgangp_stage(generator, critic, train_loader, config, tracker=None):
    """
    Stage 1: Conditional WGAN-GP learns the class-wise distribution of gear fault data.
    """
    device = config["device"]
    training_dir = _get_training_dir(config)

    # Removed pinn_criterion initialization

    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))

    critic_loss_history = []
    generator_loss_history = []
    gp_history = []
    wgan_total_loss_history = []

    print("\nStarting WGAN-GP Training Phase...")
    global_step = 0
    for epoch in range(config["epochs_stage_1"]):
        epoch_g_loss, epoch_c_loss = 0.0, 0.0
        g_steps, c_steps = 0, 0

        for i, (real_x, labels) in enumerate(train_loader):
            real_x = real_x.to(device)
            labels = labels.to(device)
            batch_size = real_x.size(0)

            # ---------------------
            #  Train Critic
            # ---------------------
            opt_c.zero_grad()
            z = torch.randn(batch_size, config["latent_dim"], device=device)
            fake_x = generator(z, labels)

            real_validity = critic(real_x, labels)
            fake_validity = critic(fake_x.detach(), labels)
            gp = compute_gradient_penalty(critic, real_x, fake_x.detach(), labels, device)

            # Eq 2: L_D = -E[D(x,y)] + E[D(G(z,y),y)] + lambda_gp * L_gp
            loss_c = -torch.mean(real_validity) + torch.mean(fake_validity) + config["lambda_gp"] * gp
            loss_c.backward()
            opt_c.step()

            epoch_c_loss += loss_c.item()
            c_steps += 1

            critic_loss_history.append((global_step, loss_c.item()))
            gp_history.append((global_step, gp.item()))
            wgan_total_loss_history.append((global_step, loss_c.item()))

            # ---------------------
            #  Train Generator
            # ---------------------
            if i % 5 == 0:
                opt_g.zero_grad()
                z = torch.randn(batch_size, config["latent_dim"], device=device)
                fake_x = generator(z, labels)
                fake_validity = critic(fake_x, labels)

                # Eq 5: L_G = -E[D(G(z,y),y)]
                loss_g = -torch.mean(fake_validity)

                loss_g.backward()
                opt_g.step()

                epoch_g_loss += loss_g.item()
                g_steps += 1

                generator_loss_history.append((global_step, loss_g.item()))

            global_step += 1

        avg_c_loss = epoch_c_loss / max(c_steps, 1)
        avg_g_loss = epoch_g_loss / max(g_steps, 1)
        print(
            f"[WGAN Stage 1] Epoch {epoch+1}/{config['epochs_stage_1']} | "
            f"D Loss: {avg_c_loss:.4f} | G Loss: {avg_g_loss:.4f}"
        )

        if tracker:
            tracker.log_metrics(
                {"critic_loss": avg_c_loss, "generator_loss": avg_g_loss},
                step=epoch + 1,
                prefix="WGAN_Stage",
            )

    # Export plots
    if critic_loss_history:
        x_c, y_c = zip(*critic_loss_history)
        _save_line_plot(
            x_c, y_c,
            "Critic Loss over Iterations",
            "Iteration", "Critic Loss",
            os.path.join(training_dir, "wgan_critic_loss_iterations.png"),
        )

    if generator_loss_history:
        x_g, y_g = zip(*generator_loss_history)
        _save_line_plot(
            x_g, y_g,
            "Generator Loss over Iterations",
            "Iteration", "Generator Loss",
            os.path.join(training_dir, "wgan_generator_loss_iterations.png"),
        )

    if gp_history:
        x_gp, y_gp = zip(*gp_history)
        _save_line_plot(
            x_gp, y_gp,
            "WGAN-GP Gradient Penalty over Iterations",
            "Iteration", "Gradient Penalty",
            os.path.join(training_dir, "wgan_gp_iterations.png"),
        )

    if wgan_total_loss_history:
        x_w, y_w = zip(*wgan_total_loss_history)
        _save_line_plot(
            x_w, y_w,
            "WGAN Critic Total Loss over Iterations",
            "Iteration", "Total Critic Loss",
            os.path.join(training_dir, "wgan_total_loss_iterations.png"),
        )


def train_classifier_stage(generator, classifier, train_loader, config, model_name="Classifier", augment=False, tracker=None):
    """
    Stage 2: Classifier is trained using the cross-entropy loss with real and augmented data.
    """
    device = config["device"]
    training_dir = _get_training_dir(config)
    opt_clf = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Eq 7: Categorical cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    if generator:
        generator.eval()

    loss_history = []
    acc_history = []

    print(f"Training Classifier Context: [{model_name}]")
    global_step = 0
    for epoch in range(config["epochs_stage_2"]):
        epoch_loss, epoch_acc, steps = 0.0, 0.0, 0

        for real_x, labels in train_loader:
            real_x = real_x.to(device)
            labels = labels.to(device)
            batch_size = real_x.size(0)

            # Utilize generated samples to enrich the training set
            if generator and augment:
                z = torch.randn(batch_size, config["latent_dim"], device=device)
                fake_x = generator(z, labels)
                combined_x = torch.cat([real_x, fake_x], dim=0)
                combined_labels = torch.cat([labels, labels], dim=0)
            else:
                combined_x = real_x
                combined_labels = labels

            opt_clf.zero_grad()
            logits = classifier(combined_x)

            loss_cls = criterion(logits, combined_labels)
            loss_cls.backward()
            opt_clf.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == combined_labels).float().mean().item()

            epoch_loss += loss_cls.item()
            epoch_acc += acc
            steps += 1

            loss_history.append((global_step, loss_cls.item()))
            acc_history.append((global_step, acc))
            global_step += 1

        avg_loss = epoch_loss / max(steps, 1)
        avg_acc = epoch_acc / max(steps, 1)
        print(
            f"  [{model_name}] Epoch {epoch+1}/{config['epochs_stage_2']} | "
            f"Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}"
        )

        if tracker:
            tracker.log_metrics(
                {"loss": avg_loss, "acc": avg_acc},
                step=epoch + 1,
                prefix=model_name.replace(" ", "_"),
            )

    # Export plots
    if loss_history:
        x_l, y_l = zip(*loss_history)
        _save_line_plot(
            x_l, y_l,
            f"{model_name} Train Loss over Iterations",
            "Iteration", "Loss",
            os.path.join(training_dir, f"{model_name.lower().replace(' ', '_')}_train_loss_iterations.png"),
        )

    if acc_history:
        x_a, y_a = zip(*acc_history)
        _save_line_plot(
            x_a, y_a,
            f"{model_name} Train Accuracy over Iterations",
            "Iteration", "Accuracy",
            os.path.join(training_dir, f"{model_name.lower().replace(' ', '_')}_train_acc_iterations.png"),
        )