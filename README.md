🌧️ Input–Output Nowcasting Evaluation

This repository provides code for the paper:

\*\*"An Explainable Framework for Optimizing Input–Output Configuration in Deep Learning-Based Rainfall Nowcasting"\*\*

It evaluates rainfall nowcasting models with varying input–output configurations using deep learning, focusing on explainability and performance under different temporal settings.

---

📁 Project Structure

\- `run\_predictions.py` – Main evaluation script  

\- `utils.py` – Utilities for data loading, scaling, plotting, and model inference  

\- `metrics/` – Contains MAE, RMSE, and CSI metric functions  

\- `nowcast\_unet/` – U-Net-based model architecture (2D CNN)  

\- `checkpoints/` – Pretrained model weights organized by input configuration  

\- `sample\_data/` – Example radar sequences for testing  

\- `sequence\_builder.py` – Functions for building radar input/output sequences  

\- `README.md` – Project documentation
---

💻 Setup & Requirements

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HydroAI/InputOutputNowcasting.git
   cd InputOutputNowcasting

2. Install required  packages:

You can use either pip or conda:
🔹 Option A – Using pip
bash
pip install -r requirements.txt

🔹 Option B – Using conda (recommended for reproducibility)
bash
conda env create -f environment.yml
conda activate nowcast-env

3. Add model checkpoints
Download pretrained model weights and place them into the appropriate folders:

For example:
- `checkpoints/2in/weights-best.pth`
- `checkpoints/4in/weights-best.pth`
- ...
- `checkpoints/12in/weights-best.pth`

> 🔎 Note: The code is already included via the cloned repository.

bash
pip install -r requirements.txt

3. Prepare model checkpoints:
Download or copy pretrained model weights into checkpoints/{N}in/weights-best.pth for each input configuration (e.g., 2in, 3in, etc.).

🚀 Usage

\### 1. Run the evaluation script

```bash

python run\_predictions.py

This runs inference and evaluation on the radar sequences for all selected models and saves outputs to the outputs/ folder.


⚙️ Input–Output Configuration
Inside run_predictions.py, configure:

Input frame settings using model_config:
model_config = {
    2: {"slice": slice(22, 24), ...},
    3: {"slice": slice(21, 24), ...},
    ...
}

Output setting using:
n_output_frames = 1  # Options: 1, 6, or 12

Where:

1 → Recursive prediction (1 frame at a time, 12 steps)
6 → Predict 6 frames + recursively predict next 6
12 → Full 12-frame prediction in one forward pass

📊 Outputs and Visualization

The script generates:

✅ .gif animations: Observed vs. predicted rainfall
✅ Grid plots comparing multiple model predictions
✅ MAE, RMSE, and CSI scores over time
✅ Thresholded CSI plots for multiple rain rate thresholds

All outputs are saved to the outputs/ directory.


📄 License

This project is open-source under the MIT License. See LICENSE for details.

📬 Contact

For questions or collaborations, contact:

Ahmed Abdelhalim

University of Bristol

✉️ ahmed.abdelhalim@bristol.ac.uk

✉️ ahmed\_abdelhalim@mu.edu.eg

🔗 https://github.com/Ahmed-HydroAI





