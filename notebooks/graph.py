import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

model_name = "model-1763799751"


def create_acc_loss_graph(model_name):
    content = open("model.log", "r").read()

    entries = content.split(model_name)[1:]

    times = []
    accuracies = []
    losses = []
    val_accs = []
    val_losses = []

    for entry in entries:
        parts = entry.lstrip(',').split(',')

        if len(parts) >= 5:
            try:
                timestamp = float(parts[0].strip())
                acc = float(parts[1].strip())
                loss = float(parts[2].strip())
                val_acc = float(parts[3].strip())
                val_loss = float(parts[4].strip())

                times.append(timestamp)
                accuracies.append(acc)
                losses.append(loss)
                val_accs.append(val_acc)
                val_losses.append(val_loss)
            except ValueError as e:
                print(f"Skipping entry due to error: {e}")
                continue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(times, accuracies, label='Training Accuracy', marker='o', markersize=3)
    ax1.plot(times, val_accs, label='Validation Accuracy', marker='o', markersize=3)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy over Time')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times, losses, label='Training Loss', marker='o', markersize=3)
    ax2.plot(times, val_losses, label='Validation Loss', marker='o', markersize=3)
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


create_acc_loss_graph(model_name)