from ultralytics import YOLO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Initialize the YOLO model
    model = YOLO('models/test3.pt')

    # Path to your data configuration YAML
    data_yaml_path = 'C:/Users/thoba/AiProject/CS4100-Final-Project/ProjectPredictiveComputerVision/Project/Valorant Object Detection.v22i.yolov8/data.yaml'

    # Train the model and capture statistics
    try:
        print('Training model...')
        results = model.train(data=data_yaml_path, epochs=1, imgsz=640, device='cuda', amp=True)

        # Save the model after training
        model.save('models/tesetforPlotting.pt')

        # Extract loss and mAP from training results
        losses = results.history['metrics/loss']
        mAPs = results.history.get('metrics/mAP_0.5', [])

        print('Training completed.')

    except RuntimeError as e:
        print("Error encountered during training:")
        print(e)
        losses, mAPs = [], []  # Ensure lists are initialized

    # Evaluate the model on the validation dataset
    print("Validating model...")
    val_results = model.val(data=data_yaml_path)
    print("Validation results:", val_results)

        # Safely attempt to access mAP_0.5
    if hasattr(val_results, 'metrics'):
        mAP = val_results.metrics.get('mAP_0.5', None)
        if mAP is not None:
            mAPs.append(mAP)
        else:
            print("mAP metric not found in validation results.")
    else:
        print("No metrics attribute found in val_results.")

    # Plot the loss and mAP over the epochs
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Training Loss')
    if mAPs:
        plt.plot(mAPs, label='mAP')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Statistics')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig('training_stats.png')
    print("Training statistics plot saved as 'training_stats.png'.")

    # Show the plot
    plt.show()
