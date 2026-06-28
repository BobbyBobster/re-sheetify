# 📚 Re-sheetify

Ground-up rebuild of the [Sheetify project](https://github.com/emilegeagea/sheetify).
This rebuild is meant to make maintenance of the project easier. Hence some components are re-architected and changed.
Also this will allow us to more easily train different instances of the model to compare performance.

<img src="/../screenshots/screenshots/streamlit-output.png" alt="Sheetify webapp in use"/>
<br>
App home: https://sheetify-frontend-614562267249.us-west1.run.app


## Sheetify info
> A deep learning automatic music transcription system.

Sheetify is an end-to-end automatic music transcription system that converts piano audio recordings into MIDI files and printable sheet music. Inspired by Google's **Onsets and Frames** architecture, Sheetify uses deep neural networks to detect note onsets, pitches, and durations from raw audio, enabling accurate transcription of polyphonic piano performances.

The model was trained and evaluated using the **MAESTRO dataset**, a large-scale collection of piano performances with precisely aligned audio recordings and MIDI annotations, providing high-quality ground truth data for automatic music transcription.




## Getting Started
### Setup
Use your preferred Python package manager to install the `requirements.txt` file.

### ENV Variables
Create `.env` file
```
touch .env
```
Inside `.env`, set these variables. For any APIs, see group Slack channel.
```
# In case you want to run your own GCP services/tasks
GCP_PROJECT=gcp_project_to_deploy_to
GCP_REGION=gcp_region_to_deploy_to
BUCKET_NAME=gcs_bucket_to_store_models_and_data
TRAINER_IMAGE=image_name
TRAINER_MEMORY=16Gi
FRONTEND_IMAGE=image_name
FRONTEND_MEMORY=4Gi
```


## Built With
- [Tensorflow](https://www.tensorflow.org/) - ML platform
- [Librosa](https://librosa.org/) - Audio processing
- [PrettyMIDI](https://github.com/craffel/pretty-midi) - MIDI processing
- [LilyPond](https://lilypond.org/) - Sheet music generation

## Acknowledgements
Sheetify itself is a re-implementation of the ideas found in:
> Hawthorne, C., Elsen, E., Song, J., Roberts, A., Simon, I., Raffel, C., Engel, J., Oore, S., & Eck, D. (2018).
> **Onsets and Frames: Dual-Objective Piano Transcription**
> Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR).

## License
This project is licensed under the MIT License
