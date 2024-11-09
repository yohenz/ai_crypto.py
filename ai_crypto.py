import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Konfigurasi dan variabel global
win_count = 0
lose_count = 0
total_signals = 0

# Fungsi untuk mengupdate statistik
def update_stats(is_win):
    global win_count, lose_count
    if is_win:
        win_count += 1
    else:
        lose_count += 1

# Fungsi untuk menghitung winrate
def get_winrate():
    if win_count + lose_count == 0:
        return 0
    return (win_count / (win_count + lose_count)) * 100

# Mengambil data harga BTC/USDT dari TradingView melalui yfinance
def get_data():
    data = yf.download("BTC-USD", interval="5m", period="7d")
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    return data.dropna()

# Membuat model AI untuk prediksi
def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Mengambil sinyal dari model AI
def generate_signal(data, model):
    global total_signals
    features = data[['RSI', 'MACD', 'MACD_signal']].values
    predictions = model.predict(features)
    last_price = data.iloc[-1]['Close']
    
    # Buy jika prediksi > 0.6, Short jika prediksi < 0.4
    if predictions[-1] > 0.6:
        total_signals += 1
        return ("BUY", last_price, last_price * 0.98, last_price * 1.02)  # Entry, Stop Loss, Target
    elif predictions[-1] < 0.4:
        total_signals += 1
        return ("SHORT", last_price, last_price * 1.02, last_price * 0.98)  # Entry, Stop Loss, Target
    return None

# Fungsi utama untuk menjalankan analisis
def main():
    global win_count, lose_count
    data = get_data()
    model = build_model()

    # Melatih model (gunakan data historis untuk latih)
    X = data[['RSI', 'MACD', 'MACD_signal']].values
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # Mengambil sinyal terbaru
    signal = generate_signal(data, model)
    if signal:
        action, entry_price, stop_loss, target_price = signal
        print(f"Sinyal: {action}")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Stop Loss: ${stop_loss:.2f}")
        print(f"Target Price: ${target_price:.2f}")

        # Simulasi apakah sinyal berhasil
        current_price = data.iloc[-1]['Close']
        if action == "BUY" and current_price >= target_price:
            update_stats(True)
        elif action == "SHORT" and current_price <= target_price:
            update_stats(True)
        else:
            update_stats(False)

        print(f"Total Sinyal: {total_signals}")
        print(f"Win: {win_count}, Lose: {lose_count}")
        print(f"Winrate: {get_winrate():.2f}%")

if __name__ == "__main__":
    main()