# SoccerSum

## Source Code
- **Repository**: [SoccerSum GitHub](https://github.com/simula/SoccerSum)
- **Programming Languages**: Python, JavaScript, HTML, CSS, AJAX
- **Python Version**: Python 3.11

## Packages and Libraries
- **Required Python Packages**: `flask`, `re`, `torch`, `torchvision`, `ultralytics`, `cv2`, `numpy`, `requests`, `openai`, `whisper`, `librosa`, `sklearn.feature_extraction.text`, `sklearn.metrics.pairwise`, `matplotlib`, `mpl_toolkits.mplot3d`

## Dataset
The SoccerSum dataset was curated by capturing and annotating soccer videos from the Norwegian Eliteserien league. This collection spans three years, covering 2021, 2022, and 2023. It comprises 750 frames from 41 unique sequences, with 4 to 40 frames per sequence, carefully chosen to represent a diverse selection of scenarios encountered in professional soccer games.
- **Zenodo Record**: [SoccerSum dataset on Zenodo](https://zenodo.org/records/10612084)

## Hardware Requirements
- **System Tested On**: Linux
- **GPU Requirement**: At least 1 GPU with 4 GB VRAM

## Installation and Experimentation

### Setup
1. Clone the SoccerSum repository:
   ```
   git clone https://github.com/simula/SoccerSum
   ```

2. Install the necessary requirements:
   ```
   cd SoccerSum
   pip install -r requirements.txt
   ```

3. Download the model weights for Detection, Segmentation, and Classification: [SoccerSum HuggingFace repository](https://huggingface.co/SimulaMet-HOST/SoccerSum)

4. Place the weights in the `weights` folder. Adjust paths in `main.py`.

### Running the Application
1. Ensure port 5000 is free on your machine.

2. Start the Flask application:
   ```
   python3 app.py
   ```

3. The GUI will open in your default web browser.

### Usage
- Provide the path to a Goal event file (MP4 or M3U8 URL).
- A valid OPENAI API Access Token for GPT-4 is required.
- The system processes a 30-second clip in about 1 minute; longer clips take more time.
- Progress is shown on the GUI sidebar.


## Citations

Please cite our research using the following BibTeX entries:

<pre><code>
@incollection{Houshmand_MMSYS_ODS,
  author = {Houshmand Sarkhoosh, Mehdi and Midoglu, Cise and Shafiee Sabet, Saeed and Halvorsen, P{\aa}l},
  title = {{The SoccerSum Dataset for Automated Detection, Segmentation, and Tracking of Objects on the Soccer Pitch}},
  booktitle = {{MMSys'24 : The 15th ACM Multimedia Systems Conference}},
  year = {2024},
  month = apr,
  date = {2024-04-15},
  urldate = {2024-04-15},
  isbn = {979-8-4007-0412-3/24/04},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3625468.3652180}
}
</code></pre>

<pre><code>
@incollection{Houshmand_MMSYS_demo,
  author = {Houshmand Sarkhoosh, Mehdi and Midoglu, Cise and Shafiee Sabet, Saeed and Halvorsen, P{\aa}l},
  title = {{Multimodal AI-Based Summarization and Storytelling for Soccer on Social Media}},
  booktitle = {{MMSys'24 : The 15th ACM Multimedia Systems Conference}},
  year = {2024},
  month = apr,
  date = {2024-04-15},
  urldate = {2024-04-15},
  isbn = {979-8-4007-0412-3/24/04},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3625468.3652197}
}
</code></pre>

You can refer to the `CITATION.cff` file provided in the root directory if you use the software in this repository.

