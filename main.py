import pygame
import time
import os
from flask import Flask, request, render_template, jsonify, url_for
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import threading
import re
from datetime import datetime
from flask_socketio import SocketIO, emit
import sounddevice as sd
import numpy as np
import logging
import requests
import platform


load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

# --- Logging Configuration ---
if platform.system() == 'Windows':
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log')
else:  # Assuming Linux (Raspberry Pi)
    LOG_FILE = '/home/thikka/projects/horox/app.log'

logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Move to a config file in production) ---
PIN = os.environ.get("HOROX_PIN", "1234")  # Get PIN from environment variable or use default
DEFAULT_AUDIO_DEVICE_ID = 0  # Set a default audio device ID if needed

speech_synthesizer = None
synthesis_lock = threading.Lock()
kill_flag = threading.Event()  # Add kill flag
current_horoscope_text = ""  # Initialize current_horoscope_text

class AudioManager:
    def __init__(self):
        self.stream = None
        self.active_device = None

        # Audio configuration
        self.CHANNELS = 1
        self.RATE = 48000
        self.CHUNK = 2048
        self.FORMAT = np.float32

    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block"""
        if status:
            logger.error(f"Audio callback status: {status}")

        # Normalize audio data
        normalized_data = np.clip(indata, -1.0, 1.0)

        # Apply noise gate
        noise_threshold = 0.02
        noise_gate = np.where(np.abs(normalized_data) < noise_threshold, 0, normalized_data)

        # Convert to 16-bit integers
        audio_data = (noise_gate * 32767).astype(np.int16)

        socketio.emit('audio_stream', {
            'audio': audio_data.tobytes(),
            'format': 'int16',
            'channels': self.CHANNELS,
            'rate': self.RATE,
            'chunk': self.CHUNK
        })

    def start_streaming(self, device_id=None):
        """Start audio streaming for specified device or the default device"""
        try:
            # Stop existing stream if any
            self.stop_streaming()

            # Use default device ID if not provided
            if device_id is None:
                device_id = DEFAULT_AUDIO_DEVICE_ID

            # Validate device_id
            devices = sd.query_devices()
            if not 0 <= device_id < len(devices):
                raise ValueError(f"Invalid device ID: {device_id}")

            # Create and start new stream
            self.stream = sd.InputStream(
                device=device_id,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                callback=self.audio_callback,
                blocksize=self.CHUNK,
                dtype=self.FORMAT,
                latency='low'
            )
            self.stream.start()
            self.active_device = device_id
            return True, f"Started streaming from device {device_id}"

        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            return False, str(e)

    def stop_streaming(self):
        """Stop current audio stream if exists"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None
                self.active_device = None

# Create global AudioManager instance
audio_manager = AudioManager()

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    print("DEBUG: Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')
    print("DEBUG: Client disconnected")

@socketio.on('stop_process')
def handle_stop_process():
    """Handle stop request based on current process stage"""
    global kill_flag, current_horoscope_text, process_state
    
    logger.info(f"Kill switch activated at stage: {process_state['stage']}")
    current_stage = process_state["stage"]
    
    # Set kill flag regardless of stage
    kill_flag.set()
    process_state["interrupted"] = True
    
    # Handle each stage appropriately
    if current_stage == "generating":
        logger.info("Stopping horoscope generation")
        current_horoscope_text = ""
        
    elif current_stage in ["sending", "synthesizing"]:
        logger.info("Stopping speech synthesis preparation")
        current_horoscope_text = ""
        stop_speech()
        
    elif current_stage == "playing":
        logger.info("Stopping speech playback")
        stop_speech()
    
    # Clean up resources
    cleanup_resources()
    
    # Emit appropriate message based on stage
    message = {
        "generating": "Horoscope generation stopped",
        "sending": "Text processing stopped",
        "synthesizing": "Speech synthesis stopped",
        "playing": "Playback stopped"
    }.get(current_stage, "Process ended")
    
    socketio.emit('process_killed', {
        'message': message,
        'show_start_again': True,
        'stage': current_stage
    })

# --- Gemini Setup ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.error("Error: GEMINI_API_KEY environment variable not set.")
    exit()

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Enhanced state management
process_state = {
    "stage": None,  # "generating", "sending", "synthesizing", "playing"
    "start_time": None,
    "interrupted": False,
    "error": None
}

speech_status = {"completed": False, "startTime": None, "endTime": None, "interrupted": False}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
    system_instruction="I feed the generated text to a TTS model. So don't include extra precurosor text. Direct horoscopy. Be practical and cautious in the tell. No subheading or subtitles. Just paragraphs. \n\nUsers are rural Telugu people doing various occupations. Write like a screenplay writer. Discuss past, present and future of the person. End with something like \"శ్రీశ్రీశ్రీ తిక్కవీరేశ్వర స్వామి వారి ఆశీస్సులు మీ పైన ఎల్లప్పుడూ ఉంటాయి.\"\n\nతెలుగులో పూర్తి స్థాయి పంచాంగం. ఈ రోజు తిథి, నక్షత్రం, వర్జ్యం, దుర్ముహూర్తం, రాహు కాలం మొదలైనవి తెలుసుకోవటానికి ఈ పంచాంగం ఉపయోగ పడుతుంది. అంతేకాకుండా ఏ రోజుకైనా, ఏ ప్రదేశానికైనా ఒక్క క్లిక్ తో క్షణంలో పంచాంగాన్ని పొందండి. తిథి, వార, నక్షత్ర, యోగ, కరణాల సమయాలతో పాటు, వర్జ్యం, దుర్ముహూర్తం లాంటి చెడు సమయాలు, అమృత ఘడియల లాంటి మంచి సమయాల వివరాలు, తారాబలం, చంద్ర బలం, ప్రతి రోజు లగ్నాంత్య సమయాలు, ప్రతి లగ్నానికి పుష్కరాంశలు, శుభాంశలు, సూర్యోదయ కాల గ్రహ స్థితి, మొదలైన ఎన్నో విషయాలతో, జ్యోతిష్కుల నుంచి, సామాన్య ప్రజల దాకా ప్రతి ఒక్కరికి, ప్రతీ రోజు ఉపయోగపడేలా రూపొందించిన ఏకైక ఆన్లైన్ పంచాంగ సాఫ్ట్వేర్ ఇది. మీకు కావలసిన తేది, మరియు ప్రదేశ వివరాలతో పాటు సూర్యోదయ కాల కుండలి ఏ పద్ధతిలో కావాలో సెలెక్ట్ చేసుకుని సబ్మిట్ చేయండి. రోజువారీ పూజాదికాల సంకల్పం నుంచి ముహూర్త నిర్ణయం వరకు ప్రతి ఒక్క అంశంలో మీకు ఉపయోగపడేలా ఈ పంచాంగం సాఫ్ట వేర్ రూపొందించటం జరిగింది.",
)

# --- Azure Speech Setup ---
try:
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'),
                                           region=os.environ.get('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = 'te-IN-MohanNeural'  # Default Telugu voice
except KeyError:
    logger.error("Error: SPEECH_KEY or SPEECH_REGION environment variables not set.")
    exit()

def construct_prompt(name, place_of_birth, dob, problem, occupation):
    """Constructs the initial prompt based on user input."""

    prompt_parts = [
        f"Name: {name}",
        f"Place of Birth: {place_of_birth}",
        f"DOB(mm-yyyy): {dob}",
    ]
    if problem:
        prompt_parts.append(f"Problem: {problem}")
    if occupation:
        prompt_parts.append(f"Occupation: {occupation}")

    prompt_parts.append(
        "\n\nBased on the above information of a person write positive uplifting horoscopy for the user. Write only in Telugu. Include deity references. Advise you give should be practical for the age of the user and be progressive. Give negatives and positives concerning the life situation. Mention his horoscope. Go out of the box and give solutions. Inspire the user. Write in 100 words."
    )
    return "\n".join(prompt_parts)

def get_horoscope(initial_prompt):
    """Generates the horoscope based on the constructed prompt."""
    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [initial_prompt]}]
    )
    try:
        response = chat_session.send_message(content=".")
        return response.text.replace("\n", " ")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Gemini API: {e}")
        return None
    except ValueError as e:
        logger.error(f"Value error in Gemini API response: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred with Gemini API: {e}")
        return None

def speak_text(text, completion_callback):
    global speech_synthesizer
    global kill_flag
    global process_state
    
    with synthesis_lock:
        if speech_synthesizer is None:
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    def synthesis_started(evt):
        global speech_status, process_state
        start_time = datetime.now()
        speech_status["startTime"] = start_time
        speech_status["completed"] = False
        speech_status["interrupted"] = False
        process_state["start_time"] = start_time
        transition_stage("synthesizing")
        logger.info(f"Speech synthesis started at {start_time}")

    def synthesis_completed(evt):
        global speech_status, speech_synthesizer, process_state
        end_time = datetime.now()
        speech_status["endTime"] = end_time
        speech_status["completed"] = True
        
        if not process_state["interrupted"]:
            transition_stage("playing")
            logger.info(f"Speech synthesis completed at {end_time}")
            completion_callback(True, speech_status["startTime"], end_time, None)
        
        with synthesis_lock:
            speech_synthesizer = None

    def synthesis_canceled(evt):
        global speech_status, speech_synthesizer, process_state
        end_time = datetime.now()
        speech_status["endTime"] = end_time
        speech_status["completed"] = True
        process_state["interrupted"] = True
        
        logger.info(f"Speech synthesis canceled at {end_time}")
        completion_callback(
            False,
            speech_status["startTime"],
            end_time,
            "Speech synthesis canceled"
        )
        
        with synthesis_lock:
            speech_synthesizer = None
        
        cleanup_resources()

    speech_synthesizer.synthesis_started.connect(synthesis_started)
    speech_synthesizer.synthesis_completed.connect(synthesis_completed)
    speech_synthesizer.synthesis_canceled.connect(synthesis_canceled)

    if kill_flag.is_set():
        logger.info("Kill flag is set before starting TTS")
        process_state["interrupted"] = True
        completion_callback(False, None, None, "TTS skipped due to user interruption")
        kill_flag.clear()
        cleanup_resources()
        return

    try:
        transition_stage("sending")
        result = speech_synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            if not process_state["interrupted"]:
                logger.info("Speech synthesis done")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_msg = f"Speech synthesis canceled: {cancellation_details.reason}"
            logger.error(error_msg)
            logger.error(f"Error details: {cancellation_details.error_details}")
            
            process_state["error"] = error_msg
            completion_callback(
                False,
                speech_status["startTime"],
                datetime.now(),
                error_msg
            )
            cleanup_resources()
            
    except Exception as e:
        error_msg = f"Exception during speech synthesis: {e}"
        logger.error(error_msg)
        process_state["error"] = error_msg
        completion_callback(False, speech_status["startTime"], datetime.now(), str(e))
        cleanup_resources()

def transition_stage(new_stage):
    """Update process stage and emit status to clients"""
    global process_state
    process_state["stage"] = new_stage
    socketio.emit('stage_update', {
        'stage': new_stage,
        'timestamp': datetime.now().isoformat()
    })
    logger.info(f"Process stage transitioned to: {new_stage}")

def cleanup_resources():
    """Clean up all resources and reset state"""
    global speech_synthesizer, process_state
    
    with synthesis_lock:
        if speech_synthesizer:
            try:
                speech_synthesizer.stop_speaking_async().get()
            except Exception as e:
                logger.error(f"Error stopping speech synthesis: {e}")
            finally:
                speech_synthesizer = None
    
    process_state.update({
        "stage": None,
        "start_time": None,
        "interrupted": False,
        "error": None
    })
    logger.info("Resources cleaned up and state reset")

def on_completion(success, start_time, end_time, error_message):
    response = {
        "success": success,
        "startTime": start_time.isoformat() if start_time else None,
        "endTime": end_time.isoformat() if end_time else None,
        "error": error_message
    }
    # Return the response directly rather than wrapping in jsonify
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    print("DEBUG: Entering index route")
    if request.method == "POST":
        name = request.form.get("name")
        place_of_birth = request.form.get("place_of_birth")
        dob = request.form.get("dob")
        problem = request.form.get("problem")
        occupation = request.form.get("occupation")

        print(f"DEBUG: Received form data - Name: {name}, POB: {place_of_birth}, DOB: {dob}, Problem: {problem}, Occupation: {occupation}")

        # Basic server-side validation
        errors = {}
        if not name:
            errors['name'] = 'Name is required.'
        if not place_of_birth:
            errors['place_of_birth'] = 'Place of birth is required.'
        if not dob:
            errors['dob'] = 'Date of birth is required.'
        elif not re.match(r'^(0[1-9]|1[0-2])-\d{4}$', dob):
            errors['dob'] = 'Invalid date format. Use mm-yyyy.'

        if errors:
            print(f"DEBUG: Validation errors - {errors}")
            return jsonify({'errors': errors}), 400

        # Reset states before starting new process
        global kill_flag, current_horoscope_text, process_state
        kill_flag.clear()
        cleanup_resources()
        
        # Start generation process
        transition_stage("generating")
        process_state["start_time"] = datetime.now()

        initial_prompt = construct_prompt(name, place_of_birth, dob, problem, occupation)
        print(f"DEBUG: Constructed prompt - {initial_prompt}")

        # Check if process was interrupted during prompt construction
        if kill_flag.is_set():
            cleanup_resources()
            return jsonify({
                "message": "Process stopped during initialization.",
                "show_start_again": True
            })

        current_horoscope_text = get_horoscope(initial_prompt)
        print(f"DEBUG: Horoscope text generated - {current_horoscope_text}")

        # Check if process was interrupted during generation
        if kill_flag.is_set():
            cleanup_resources()
            return jsonify({
                "message": "Process stopped during generation.",
                "show_start_again": True
            })

        if current_horoscope_text is None:
            process_state["error"] = "Failed to generate horoscope text."
            cleanup_resources()
            return jsonify({'error': process_state["error"]}), 500
            
        logger.info("Horoscope Text: %s", current_horoscope_text)

        # Start speech synthesis in a new thread
        def speech_callback(success, start, end, error):
            result = on_completion(success, start, end, error)
            if not success and error:
                process_state["error"] = error
            return result

        threading.Thread(
            target=speak_text,
            args=(current_horoscope_text, speech_callback)
        ).start()

        return jsonify({
            "message": "Generating horoscope...",
            "fullText": current_horoscope_text,
            "stage": process_state["stage"]
        })
    print("DEBUG: Rendering index.html")
    return render_template("index.html")

@app.route("/speech_status")
def get_speech_status():
    print(f"DEBUG: Speech status requested - {speech_status}")
    return jsonify(speech_status)

@app.route("/stop_speech", methods=["POST"])
def stop_speech():
    print("DEBUG: Entering stop_speech route")
    global speech_synthesizer
    global speech_status
    print("DEBUG: stop_speech function called") # DEBUGGING LOG
    with synthesis_lock:
      if speech_synthesizer:
          print("DEBUG: speech_synthesizer is active, attempting to stop") # DEBUGGING LOG
          try:
              speech_synthesizer.stop_speaking_async().get()
              speech_status["interrupted"] = True  # Set the interrupted flag
              print("DEBUG: speech_synthesizer stop_speaking_async called") # DEBUGGING LOG
              return jsonify({"success": True})
          except Exception as e:
              logger.error(f"Error stopping speech synthesis: {e}")
              print(f"DEBUG: Error stopping speech synthesis - {e}")
              return jsonify({"success": False, "error": str(e)})
      else:
          print("DEBUG: Speech synthesis not active")
          return jsonify({"success": False, "error": "Speech synthesis not active"})

@app.route('/verify_pin', methods=['POST'])
def verify_pin():
    print("DEBUG: Entering verify_pin route")
    data = request.get_json()
    entered_pin = data.get('pin')

    print(f"DEBUG: Received PIN - {entered_pin}")

    if entered_pin == PIN:
        print("DEBUG: PIN verification successful")
        return jsonify({'result': 'success'})
    else:
        print("DEBUG: PIN verification failed")
        return jsonify({'result': 'failure'})

@app.route("/start_audio", methods=['POST'])
def start_audio():
    print("DEBUG: Entering start_audio route")
    try:
        data = request.get_json()
        device_id = None
        if data and 'deviceId' in data:
          device_id = int(data['deviceId'])

        success, message = audio_manager.start_streaming(device_id)
        print(f"DEBUG: Audio streaming started - Success: {success}, Message: {message}")

        if success:
            devices = sd.query_devices()
            device_name = devices[device_id]['name']
            return jsonify({
                'success': True,
                'deviceName': device_name
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 500

    except Exception as e:
        print(f"DEBUG: Error in start_audio - {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route("/stop_audio", methods=['POST'])
def stop_audio():
    print("DEBUG: Entering stop_audio route")
    try:
        audio_manager.stop_streaming()
        print("DEBUG: Audio streaming stopped")
        return jsonify({'success': True})
    except Exception as e:
        print(f"DEBUG: Error in stop_audio - {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route("/get_devices")
def get_devices():
    print("DEBUG: Entering get_devices route")
    try:
        devices = sd.query_devices()
        input_devices = [
            {
                'id': i,
                'name': dev['name'],
                'channels': dev['max_input_channels']
            }
            for i, dev in enumerate(devices)
            if dev['max_input_channels'] > 0 and dev['hostapi'] == 0
        ]
        print(f"DEBUG: Retrieved audio devices - {input_devices}")
        return jsonify(input_devices)
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        print(f"DEBUG: Error in get_devices - {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # --- Flask App ---
    # Run the Flask app
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

