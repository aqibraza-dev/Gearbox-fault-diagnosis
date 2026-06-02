import torch
import torch.nn as nn
import torch.autograd as autograd
from src.models.losses import PINNStyleTransferLoss

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    fake_grad = torch.ones(batch_size, 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake_grad, create_graph=True,
        retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def train_wgangp_stage(generator, critic, train_loader, config, tracker=None):
    device = config['device']
    pinn_criterion = PINNStyleTransferLoss().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))
    
    print("\nStarting WGAN-GP Training Phase...")
    for epoch in range(config['epochs_stage_1']):
        epoch_g_loss, epoch_c_loss = 0.0, 0.0
        g_steps, c_steps = 0, 0
        
        for i, (real_x, labels) in enumerate(train_loader):
            real_x, labels = real_x.to(device), labels.to(device)
            batch_size = real_x.size(0)
            
            # ---------------------
            # Train Critic
            # ---------------------
            opt_c.zero_grad()
            z = torch.randn(batch_size, config['latent_dim']).to(device)
            fake_x = generator(z, labels)
            
            real_validity = critic(real_x, labels)
            fake_validity = critic(fake_x.detach(), labels)
            gp = compute_gradient_penalty(critic, real_x, fake_x.detach(), labels, device)
            
            loss_c = -torch.mean(real_validity) + torch.mean(fake_validity) + config['lambda_gp'] * gp
            loss_c.backward()
            opt_c.step()
            epoch_c_loss += loss_c.item()
            c_steps += 1

            # ---------------------
            # Train Generator (Every 5 steps)
            # ---------------------
            if i % 5 == 0:
                opt_g.zero_grad()
                fake_x = generator(z, labels)
                fake_validity_g = critic(fake_x, labels)
                
                loss_g_adv = -torch.mean(fake_validity_g)
                loss_style_pinn = pinn_criterion(fake_x, fake_x.detach(), real_x)
                
                loss_g = loss_g_adv + (0.1 * loss_style_pinn)
                loss_g.backward()
                opt_g.step()
                epoch_g_loss += loss_g.item()
                g_steps += 1
        
        # Calculate Epoch Metrics
        avg_c_loss = epoch_c_loss / c_steps
        avg_g_loss = epoch_g_loss / (g_steps + 1e-8)
        print(f"[WGAN Stage 1] Epoch {epoch+1}/{config['epochs_stage_1']} | D Loss: {avg_c_loss:.4f} | G Loss: {avg_g_loss:.4f}")
        
        # Track parameters and run history
        if tracker:
            tracker.log_metrics({
                "critic_loss": avg_c_loss,
                "generator_loss": avg_g_loss
            }, step=epoch + 1, prefix="WGAN_Stage")

def train_classifier_stage(generator, classifier, train_loader, config, model_name="Classifier", augment=False, tracker=None):
    device = config['device']
    opt_clf = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    if generator: generator.eval()
    
    print(f"Training Classifier Context: [{model_name}]")
    for epoch in range(config['epochs_stage_2']):
        epoch_loss, epoch_acc, steps = 0.0, 0.0, 0
        for real_x, labels in train_loader:
            real_x, labels = real_x.to(device), labels.to(device)
            batch_size = real_x.size(0)
            
            if generator and augment:
                z = torch.randn(batch_size, config['latent_dim']).to(device)
                with torch.no_grad():
                    fake_x = generator(z, labels)
                combined_x = torch.cat([real_x, fake_x], dim=0)
                combined_labels = torch.cat([labels, labels], dim=0)
            else:
                combined_x, combined_labels = real_x, labels
            
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
            
        avg_loss = epoch_loss / steps
        avg_acc = epoch_acc / steps
        print(f"  [{model_name}] Epoch {epoch+1}/{config['epochs_stage_2']} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")
        
        # Track run metrics structured safely by model name identifier
        if tracker:
            tracker.log_metrics({
                "loss": avg_loss,
                "train_accuracy": avg_acc
            }, step=epoch + 1, prefix=model_name.replace(" ", "_"))