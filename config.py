# config.py
import os
from os.path import dirname, join
from dotenv import load_dotenv

# .env dosyasının bu dosya ile aynı dizinde olduğunu belirtme
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path, override=True)

# Farklı ortamlar için isimler belirleme
envs = {"development": "dev", "production": "prod"}

# Tüm konfigürasyonları tutan ana sözlük
data = {
    # Production ortamı ayarları
    envs["production"]: {
        "slack_api_token": os.getenv("SLACK_API_TOKEN"),
        "slack_channel_id": "C094GH59U4B"  
    },
    # Development ortamı ayarları
    envs["development"]: {
        "slack_api_token": os.getenv("SLACK_API_TOKEN"),
        "slack_channel_id": "C094J7WAYCE" 
    }
}

def get_config(env_args="development"): # Varsayılanı "development" yaptık
    """
    Belirtilen ortama (environment) ait konfigürasyon bilgilerini döndürür.
    """
    chosen_env = envs.get(env_args, "development")

    config_data = data[chosen_env]
    # Eski kodla uyumlu olması için 'token' ve 'channel_id' anahtarlarını ekliyoruz
    config_data['token'] = config_data.get('slack_api_token')
    config_data['channel_id'] = config_data.get('slack_channel_id')

    return config_data


# şu anlık "development" ayarlarını kullanacak şekilde sabitlendi.
SLACK_CONFIG = get_config("development")