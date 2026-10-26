"""
Client-Side Audio Deterrents & Acoustic Warning System.
Generates HTML5 Web Audio API synth sirens and Web Speech API vocal deterrents
to instantly alert candidate of cheating infractions in real time.
"""

import streamlit.components.v1 as components


def trigger_audio_siren(frequency_hz: int = 880, duration_ms: int = 600) -> None:
    """
    Synthesizes a dual-pulse square wave acoustic alert directly in client's browser.
    """
    html_code = f"""
    <script>
    (function() {{
        try {{
            var AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            var ctx = new AudioContext();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();

            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime({frequency_hz}, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime({frequency_hz * 1.5}, ctx.currentTime + {duration_ms / 1000.0});

            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + {duration_ms / 1000.0});

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start();
            osc.stop(ctx.currentTime + {duration_ms / 1000.0});
        }} catch(e) {{
            console.error('AudioContext synthesis failed: ', e);
        }}
    }})();
    </script>
    """
    components.html(html_code, height=0, width=0)


def speak_proctor_warning(message: str) -> None:
    """
    Pronounces invigilator reprimand through client-side Web Speech API.
    """
    clean_msg = message.replace('"', '\\"').replace("'", "\\'")
    html_code = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance("{clean_msg}");
            utterance.rate = 1.05;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
    }})();
    </script>
    """
    components.html(html_code, height=0, width=0)
