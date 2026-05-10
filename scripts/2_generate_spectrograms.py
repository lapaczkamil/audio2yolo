import argparse
import librosa
import os
import glob
import sys
import warnings

warnings.filterwarnings('ignore')

def generate_mel_spectrogram(audio_path, output_path, sr=44100, n_mels=128):
    """
    Konwertuje plik audio na Mel-spektrogram i zapisuje go jako obraz PNG.
    (Parametry mel jak w mel_features.py — muszą być zgodne z inferencją / ROS.)
    """
    from mel_features import SR, save_mel_png

    del n_mels
    try:
        y, sr = librosa.load(audio_path, sr=sr or SR, mono=True)
        save_mel_png(y, sr, output_path)
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")

def process_all_classes(base_input_folder, base_output_folder):
    if not os.path.exists(base_input_folder):
        print(f"Error: Input directory '{base_input_folder}' does not exist.")
        return

    class_subfolders = [f for f in os.listdir(base_input_folder) 
                        if os.path.isdir(os.path.join(base_input_folder, f))]

    if not class_subfolders:
        print(f"Warning: No subdirectories found in '{base_input_folder}'.")
        return

    print(f"Found {len(class_subfolders)} classes to process: {class_subfolders}")

    for class_name in class_subfolders:
        class_input_path = os.path.join(base_input_folder, class_name)
        class_output_path = os.path.join(base_output_folder, class_name)
        
        os.makedirs(class_output_path, exist_ok=True)
        
        wav_files = glob.glob(os.path.join(class_input_path, "*.wav"))
        file_count = len(wav_files)
        
        if file_count == 0:
            print(f"Skipping class '{class_name}': No .wav files found.")
            continue
            
        print(f"\nProcessing class '{class_name}' ({file_count} files)...")
        
        for index, audio_path in enumerate(wav_files, start=1):
            file_name = os.path.basename(audio_path)
            png_name = os.path.splitext(file_name)[0] + ".png"
            output_path = os.path.join(class_output_path, png_name)
            
            if os.path.exists(output_path):
                print(f"  [{index}/{file_count}] [Skip] {png_name} (już istnieje)")
                continue
            
            generate_mel_spectrogram(audio_path, output_path)
            print(f"  [{index}/{file_count}] Saved: {png_name}")

def _mel_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mel-spektrogramy dla YOLO (parametry jak mel_features / env AUDIO2YOLO_*).")
    p.add_argument("--input", "-i", default="2_processed_audio", help="Folder WAVów wg klas")
    p.add_argument("--output", "-o", default="3_spectrograms", help="Wyjście PNG")
    p.add_argument("--mel-n-fft", type=int, default=None, dest="mel_n_fft")
    p.add_argument("--mel-hop-length", type=int, default=None, dest="mel_hop_length")
    p.add_argument("--mel-n-mels", type=int, default=None, dest="mel_n_mels")
    p.add_argument("--mel-fmin", type=float, default=None, dest="mel_fmin")
    p.add_argument("--mel-fmax", type=float, default=None, dest="mel_fmax")
    p.add_argument("--mel-top-db", type=float, default=None, dest="mel_top_db")
    p.add_argument("--mel-preemph", type=float, default=None, dest="mel_preemph", help="0.97 typowo; nadpisuje domyślną preemfazę")
    p.add_argument("--mel-no-preemph", action="store_true", dest="mel_no_preemph")
    p.add_argument("--print-mel-config", action="store_true", help="Wypisz aktywną konfigurację mel i zakończ")
    return p


if __name__ == "__main__":
    _parser = _mel_arg_parser()
    args = _parser.parse_args()
    _scripts = os.path.dirname(os.path.abspath(__file__))
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    import mel_features as _mf

    _mf.apply_mel_argparse_args(args)
    print("Mel config:", _mf.mel_config_summary(), flush=True)
    if args.print_mel_config:
        sys.exit(0)

    process_all_classes(args.input, args.output)
    print("\nProcess completed successfully. Image dataset is ready for labeling.")