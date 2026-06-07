# -*- coding: utf-8 -*-
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tensorflow as tf
from tensorflow.keras.models import load_model



BOT_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')


MODEL_PATH = 'best_model_4.keras'

def model_propos(Y_noisy_np,model):
  import numpy as np
  X_NOISE_1=Y_noisy_np
  X_noise_all_norm = (X_NOISE_1 - np.min(X_NOISE_1)) / (np.max(X_NOISE_1) - np.min(X_NOISE_1))
  kray_1=len(X_NOISE_1)
  kray=(kray_1//2000)*2000
  def create_overlapping_windows(signal, window_size=2000, step=1300):
      windows = []
      for start in range(0, len(signal) - window_size + 1, step):
          window = signal[start:start + window_size]
          windows.append(window)
      return np.array(windows)
  step = 1300
  import numpy as np
  def overlap_add_windows(windows, window_size, step, window_func=None):
      num_windows = len(windows)
      output_length = (num_windows - 1) * step + window_size
      output_signal = np.zeros(output_length)
      window_sum = np.zeros(output_length)  
      if window_func is None:
          window = np.hanning(window_size)
      else:
          window = window_func(window_size)
      for i, win in enumerate(windows):
          start = i * step
          end = start + window_size
          output_signal[start:end] += win * window
          window_sum[start:end] += window
      epsilon = 1e-10
      output_signal = output_signal / (window_sum + epsilon)
      return output_signal
  X_noise_all_norm_1 = create_overlapping_windows(X_noise_all_norm, window_size=2000, step=step)
  X_noise_all_norm_1 = X_noise_all_norm_1[0:kray]
  X_test_1 = np.array(X_noise_all_norm_1).reshape(-1, 2000, 1)
  model_1=model
  pred_signal_1 = model_1.predict(X_test_1)
  pred_windows = pred_signal_1.squeeze()
  step = 1300
  reconstructed_signal = overlap_add_windows(pred_windows, window_size=2000, step=step)
  reconstructed_signal=reconstructed_signal[0:kray]
  pred_signal_fl_1_nm= (reconstructed_signal  - np.min(reconstructed_signal )) / (np.max(reconstructed_signal ) - np.min(reconstructed_signal ))
  return  pred_signal_fl_1_nm

# ============================================
# ФУНКЦИИ БОТА
# ============================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Привет! Отправь мне CSV-файл с данными сигнала.\n'
        'Ожидаемый формат: пробел как разделитель, данные во второй колонке.\n'
        'Я обработаю сигнал нейросетью и покажу спектры до и после.'
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.document:
        await update.message.reply_text('Пожалуйста, отправь файл в формате CSV.')
        return

    document = update.message.document
    if not document.file_name.endswith('.csv'):
        await update.message.reply_text('Пожалуйста, отправь файл в формате CSV.')
        return

    await update.message.reply_text('⏳ Получил файл, обрабатываю...')

    try:
        # Скачиваем файл_1
        file = await context.bot.get_file(document.file_id)
        file_content = io.BytesIO()
        await file.download_to_memory(file_content)
        file_content.seek(0)

        # Обработка данных
        df_noisy = pd.read_csv(file_content, sep=' ', usecols=[1], header=None)
        signal_noisy_str = df_noisy.iloc[0:50000, 0].astype(str)
        signal_noisy_processed = signal_noisy_str.str.replace(',', '.')
        signal_noisy_numeric = pd.to_numeric(signal_noisy_processed, errors='coerce').dropna()
        
        if len(signal_noisy_numeric) == 0:
            await update.message.reply_text('❌ Не удалось прочитать данные из файла')
            return

        # Подготовка сигнала
        Y_noisy_np = signal_noisy_numeric.values
        # Y_noisy_np= Y_noisy_n[0:20000]
        N = len(Y_noisy_np)
        
        # Частоты (0-4 ГГц)
        freqs = np.linspace(0, 4 * 10**9, N)
        
        # Спектр входного сигнала
        spectrum_noisy = np.abs(np.fft.fft(Y_noisy_np))
        spectrum_noisy_norm = spectrum_noisy / (np.max(spectrum_noisy[100:]) + 1e-20)
        log_spectrum_noisy_norm = np.log10(spectrum_noisy_norm + 1e-20)

        # График 1: входной сигнал
        plt.figure(figsize=(10, 5))
        freqs_mhz = freqs * 10**(-6)
        plt.plot(freqs_mhz[:N//2], log_spectrum_noisy_norm[:N//2], color='green', linewidth=0.8)
        plt.title(f'Нормированный спектр входного сигнала - {document.file_name}', fontsize=12)
        plt.xlabel('Частота (МГц)', fontsize=12)
        plt.ylabel('log10(Амплитуда)', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        img_stream1 = io.BytesIO()
        plt.savefig(img_stream1, format='png', dpi=100, bbox_inches='tight')
        img_stream1.seek(0)
        plt.close()
        
        await update.message.reply_photo(
            photo=img_stream1,
            caption=f'Входной сигнал\nДлина: {N} отсчётов\nДиапазон: 0 - 2 ГГц'
        )

       
        await update.message.reply_text('Загружаю нейросеть и обрабатываю сигнал...')
        
        if not hasattr(handle_file, 'model'):
            handle_file.model = load_model(MODEL_PATH, compile=False)
            await update.message.reply_text('Модель загружена')
        
        
        Y_out = model_propos(Y_noisy_np, model=handle_file.model)
        
       
        N_out = len(Y_out)
        freqs_out = np.linspace(0, 4 * 10**9, N_out)
        spectrum_out = np.abs(np.fft.fft(Y_out))
        spectrum_out_norm = spectrum_out / (np.max(spectrum_out[100:]) + 1e-20)
        log_spectrum_out_norm = np.log10(spectrum_out_norm + 1e-20)

        # График 2: выходной сигнал
        plt.figure(figsize=(10, 5))
        freqs_out_mhz = freqs_out * 10**(-6)
        plt.plot(freqs_out_mhz[:N_out//2], log_spectrum_out_norm[:N_out//2], color='red', linewidth=0.8)
        plt.title(f'Спектр сигнала после нейросети', fontsize=12)
        plt.xlabel('Частота (МГц)', fontsize=12)
        plt.ylabel('log10(Амплитуда)', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        img_stream2 = io.BytesIO()
        plt.savefig(img_stream2, format='png', dpi=100, bbox_inches='tight')
        img_stream2.seek(0)
        plt.close()
        
        await update.message.reply_photo(
            photo=img_stream2,
            caption=f'Выходной сигнал после нейросети\nДлина: {N_out} отсчётов\nДиапазон: 0 - 2 ГГц'
        )
        try:
            # Читаем временную метку из первой колонки CSV
            N_out = len(Y_out)
            file_content.seek(0) 
            df_noisy_time = pd.read_csv(file_content, sep=' ', usecols=[0], header=None)
            time_noisy_str_1 = df_noisy_time.iloc[0:N_out,0].astype(str)
            time_noisy_str_2 = time_noisy_str_1 .str.replace(',', '.')
            time_noisy_str =time_noisy_str_2.str.rstrip('.')
            time_noisy_numeric = pd.to_numeric(time_noisy_str, errors='coerce')
            ORIGINAL_SIGNAL_LENGTH = len(time_noisy_numeric) - 1
            TARGET_SIGNAL_LENGTH = len(time_noisy_numeric)
            time_noisy_padded = np.pad(time_noisy_numeric.values,
                                        (0, TARGET_SIGNAL_LENGTH - len(time_noisy_numeric)),
                                        'constant', constant_values=0)
      
            # Создаём DataFrame
            signal_clean=Y_out
            data_time={'время':time_noisy_padded}
            data_sig={'амплитуда':signal_clean}
            clean_DataFrame_time=pd.DataFrame(data_time)
            clean_DataFrame_sig=pd.DataFrame(data_sig)
            clean_DataFrame=pd.concat([clean_DataFrame_time,clean_DataFrame_sig], axis=1)
            # Сохраняем в CSV в памяти (не на диск)
            csv_stream = io.BytesIO()
            clean_DataFrame.to_csv(csv_stream, index=False, encoding='utf-8-sig')
            csv_stream.seek(0)  
            # Отправляем Excel файл
            await update.message.reply_document(
                 document=csv_stream,
                filename='Обработанный_Файл.csv',
                caption='Файл с обработанным сигналом'
            )
  
        except Exception as csv_error:
    # Если CSV не создался, пишем в логи, но не прерываем работу
            print(f"Ошибка при создании CSV файла: {csv_error}")
            await update.message.reply_text('⚠️ Не удалось создать CSV фаел, но графики готовы.')
        
        # === КОНЕЦ НОВОГО КОДА ===
        
    except Exception as e:
        await update.message.reply_text(f' Ошибка: {str(e)}')
    except Exception as e:
        error_msg = str(e)
        print(f"Ошибка: {error_msg}")
        await update.message.reply_text(f' Ошибка: {error_msg[:200]}')

if __name__ == '__main__':
    if not BOT_TOKEN:
        print("ОШИБКА: Не задан TELEGRAM_TOKEN в переменных окружения")
        print("Настройте Environment Variable 'TELEGRAM_TOKEN' на панели Render")
    else:
        print("Токен найден")
        print("Запускаем бота...")
        
        app = Application.builder().token(BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
        
        print(" Бот успешно запущен и ждет файлы!")
        app.run_polling()
