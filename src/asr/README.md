# ASR
Automatic Speech Recognition with Wav2vec. Supported pretrained HuggingFace models are `facebook/wav2vec2-large-960h-lv60-self` and `facebook/wav2vec2-base-960h`.

## How to run
```
pip install -r src/asr/requirements.txt
python src/asr/run.py 
```

## Example of output

```json
[{
	"audio": "/app/src/asr/data/sample.mp3",
	"transcription": [{
		"start": 0.0,
		"end": 0.8,
		"duration": 0.8,
		"label": "noEnergy"
	}, {
		"start": 0.8,
		"end": 9.84,
		"duration": 9.04,
		"label": "male",
		"transcription": ["HA MY NAME IS JOHN JWALLAM I'M A WRITER AT LARGE FOR THE NEAR TIME'S MAGAZINE AND IN TWO THOUSAND SIXTEEN I WROTE A PIECE FOR THE MAGAZINE ABOUT CLOUDS"]
	}, {
		"start": 9.84,
		"end": 10.96,
		"duration": 1.120000000000001,
		"label": "music"
	}, {
		"start": 10.96,
		"end": 14.98,
		"duration": 4.02,
		"label": "male",
		"transcription": ["YET IT'S A STORY ABOUT CLOUDS IT DOESN'T SOUND VERY IMPORTANT AND I"]
	}]
}, {
	"audio": "/app/src/asr/data/long_sample.mp3",
	"transcription": [{
		"start": 0.0,
		"end": 0.8,
		"duration": 0.8,
		"label": "noEnergy"
	}, {
		"start": 0.8,
		"end": 9.84,
		"duration": 9.04,
		"label": "male",
		"transcription": ["HA MY NAME IS JOHN JWALLAM I'M A WRITER AT LARGE FOR THE NEAR TIME'S MAGAZINE AND IN TWO THOUSAND SIXTEEN I WROTE A PIECE FOR THE MAGAZINE ABOUT CLOUDS"]
	}, {
		"start": 9.84,
		"end": 10.96,
		"duration": 1.120000000000001,
		"label": "music"
	}, {
		"start": 10.96,
		"end": 33.38,
		"duration": 22.42,
		"label": "male",
		"transcription": ["YAT IT'S A STORY ABOUT CLOUDS IT DOESN'T SOUND VERY IMPORTANT AND I GET THAT AT THIS EXACT MOMENT IN HISTORY YOU MAY NOT THINK YOU NEED TO STOP EVERYTHING AND SIT DOWN AND LISTEN TO SOME ONE TELLING A STORY ABOUT CLOUDS SO I GET THAT BUT THE CLOUDS ARE ACTUALLY KIND OF INCIDENTAL AND THAT SENSE OF TRIVIALITY THAT DISMISSIVENESS YOU MIGHT BE FEELING RIGHT NOW IS ACTUALLYAN IMPORTANT PART OF THE WHOLE"]
	}, {
		"start": 33.38,
		"end": 35.84,
		"duration": 2.460000000000001,
		"label": "music"
	}, {
		"start": 35.84,
		"end": 36.86,
		"duration": 1.019999999999996,
		"label": "noEnergy"
	}, {
		"start": 36.86,
		"end": 40.38,
		"duration": 3.520000000000003,
		"label": "male",
		"transcription": ["THERE IS ONE PERSON AT THE CENTERE OF THE STORY HIS NAME'S GAVAN PREDER PINNY"]
	}, {
		"start": 40.38,
		"end": 41.02,
		"duration": 0.6400000000000006,
		"label": "noEnergy"
	}, {
		"start": 41.02,
		"end": 50.26,
		"duration": 9.239999999999995,
		"label": "male",
		"transcription": ["GAVIN WAS LIVING IN LONDON IN TWO THOUSAND THREE WHEN ON A WHIM HE DECIDED TO LEAVE HIS JOB AND MOVE TO ROME FOR A WHILE AND WHEN HE GOT THERE HE LOOKED UP AT THE SKY AND HE SAW SOMETHING UNUSUAL"]
	}, {
		"start": 50.26,
		"end": 51.16,
		"duration": 0.8999999999999986,
		"label": "noEnergy"
	}, {
		"start": 51.16,
		"end": 51.72,
		"duration": 0.5600000000000023,
		"label": "male",
		"transcription": [""]
	}, {
		"start": 51.72,
		"end": 52.4,
		"duration": 0.6799999999999997,
		"label": "noEnergy"
	}, {
		"start": 52.4,
		"end": 58.14,
		"duration": 5.740000000000002,
		"label": "male",
		"transcription": ["THERE WEREN'T ANY CLOUDS AND HE REALIZED THAT HE MISSED THE CLOUDS THAT HE USED TO SEE IN LONDON AND HE HAD AN IDEA"]
	}, {
		"start": 58.14,
		"end": 58.56,
		"duration": 0.4200000000000017,
		"label": "noEnergy"
	}, {
		"start": 58.56,
		"end": 59.98,
		"duration": 1.4199999999999946,
		"label": "male",
		"transcription": ["HE STARTED SOMETHING CALLED THE"]
	}]
}]
```

### MP3 to WAV
Please use `convert.sh input_file [duration]` to cut and convert a mp3 input file to mono 16000 Hz wav file.
```
./convert.sh /some_path/sample.mp3
cutting /some_path/sample . mp3 to 30 seconds
converting to wav
wav file: /some_path/sample.wav
```