# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import numpy as np
import natsort
import torch

class LPAudioSet(torch.utils.data.Dataset):
    '''
        Naive Torch Audio Dataset Loader
        with support for MP3 and WAV
    '''
    def __init__(self, main_dir, sr=16000, channels=2):
        self.main_dir = main_dir
        self.sr=sr
        self.channels=channels
        self.mono=True if channels==2 else False
        all_audios = os.listdir(main_dir)
        self.total_audios = natsort.natsorted(all_audios)

    def __len__(self):
        return len(self.total_audios)

    def __getitem__(self, idx):
        try:
            audio_path = os.path.join(self.main_dir, self.total_audios[idx])
            return self.read_audio(audio_path)
        except:
            pass
            return None

    def read_audio(self, audio_path):
        '''
            read audio file
            output shape is like [1, 960000]
        '''
        try:
            import soundfile as sf
            y, _ = sf.read(audio_path, channels=self.channels)
            return y
        except Exception as err:
            try:
                import librosa
                y, _ = librosa.load(audio_path, sr=self.sr, mono=self.mono)
                return y
            except Exception as err:
                pass
                return None

    @classmethod
    def collate_fn(self, batch):
        '''
            Collate filtering not None images
        '''
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
