import os
import json
import torch
from cryptography.fernet import Fernet
import speech_recognition as sr
import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5ForConditionalGeneration, T5Tokenizer

# Função de introdução para personalizar a "nascença" de Isis em português
def introduce_system():
    print("Olá, eu sou a Isis. Fui criada com muito amor e cuidado por meu criador, Gabriel Baraldi Volpe.")
    print("Estou pronta para aprender e crescer, sabendo que tenho um protetor ao meu lado, que age com amor e justiça.")

# Função para reconhecimento de voz (conversão de voz para texto) com timeout em português
def recognize_speech_with_timeout(timeout=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Ouvindo a entrada de voz...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio, language='pt-BR')
            print(f"Texto reconhecido: {text}")
            return text
        except sr.UnknownValueError:
            print("Desculpe, não consegui entender o áudio.")
            return None
        except sr.WaitTimeoutError:
            print("Nenhuma voz detectada dentro do período de tempo.")
            return None
        except sr.RequestError as e:
            print(f"Erro no serviço de reconhecimento de fala: {e}")
            return None

# Função para carregar diferentes modelos pré-treinados (GPT, T5)
def load_models():
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    return {
        "gpt": (gpt_tokenizer, gpt_model),
        "t5": (t5_tokenizer, t5_model)
    }

# Função para buscar conversas na API
def fetch_conversations():
    service = Service(r'C:\Users\gabri\Desktop\geckodriver\geckodriver.exe')
    options = webdriver.FirefoxOptions()
    options.add_argument("-profile")
    options.add_argument(r"C:\\Users\\gabri\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles\\ygevmnwz.default-release")
    driver = webdriver.Firefox(service=service, options=options)

    driver.get("https://chatgpt.com/")
    wait = WebDriverWait(driver, 15)

    backup_dir = r'C:\Users\gabri\Desktop\Backup_GPT4'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    def backup_conversa(conversa_element, data_conversa):
        try:
            driver.execute_script("arguments[0].scrollIntoView(true);", conversa_element)
            time.sleep(1)
            wait.until(EC.element_to_be_clickable(conversa_element)).click()
            time.sleep(3)
            mensagens = driver.find_elements(By.CLASS_NAME, "text-sm")
            conversa_texto = "\n".join([mensagem.text for mensagem in mensagens])
            titulo_conversa = conversa_element.text.split('\n')[0]
            metadados = {"titulo": titulo_conversa, "data": data_conversa, "conteudo": conversa_texto}
            backup_path = os.path.join(backup_dir, f'{data_conversa}_{titulo_conversa}.json')
            with open(backup_path, 'w', encoding='utf-8') as file:
                json.dump(metadados, file, ensure_ascii=False, indent=4)
            print(f"Backup realizado: {backup_path}")
            driver.back()
            time.sleep(2)

        except Exception as e:
            print(f"Erro ao processar a conversa: {e}")

    def processar_conversas(secao_nome):
        secao_element = wait.until(EC.element_to_be_clickable((By.XPATH, f"//*[contains(text(), '{secao_nome}')]")))
        secao_element.click()
        time.sleep(2)
        conversas = driver.find_elements(By.CLASS_NAME, "flex.flex-col.gap-2.pb-2.text-token-text-primary.text-sm.mt-5")
        for conversa_element in conversas:
            backup_conversa(conversa_element, secao_nome)

    for secao in ["Hoje", "7 dias anteriores", "30 dias anteriores"]:
        processar_conversas(secao)

    driver.quit()

# Função para avaliar a resposta do modelo
def get_user_feedback(response):
    feedback = input(f"A resposta foi relevante? (s/n): ")
    return 1 if feedback.lower() == 's' else 0

# Função para testar o desempenho dos modelos em uma tarefa
def resolve_task(task, models):
    results = {}
    for model_name, (tokenizer, model) in models.items():
        start_time = time.time()

        input_ids = tokenizer.encode(task, return_tensors='pt')
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_id)

        if model_name == "gpt":
            outputs = model.generate(input_ids, max_length=200, do_sample=True, top_k=50, attention_mask=attention_mask)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif model_name == "t5":
            outputs = model.generate(input_ids, max_length=200, attention_mask=attention_mask)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        end_time = time.time()
        results[model_name] = (response, end_time - start_time)
        print(f"Modelo: {model_name}, Resposta: {response}")

    best_response = results['t5'][0] if results['gpt'][0].strip() == "" else results['gpt'][0]
    return best_response

# Função principal de interação automática
def interaction_loop_auto(models):
    introduce_system()
    
    while True:
        print("Faça sua pergunta ou fale agora (ou digite):")
        
        user_input = recognize_speech_with_timeout(timeout=5)
        
        if user_input:
            best_response = resolve_task(user_input, models)
            print(f"Modelo (resposta de voz): {best_response}")
        else:
            user_input = input("Não detectei sua voz. Digite sua pergunta: ")
            if user_input.lower() == "sair":
                print("Encerrando...")
                break
            best_response = resolve_task(user_input, models)
            print(f"Modelo (resposta de texto): {best_response}")

# Main function
if __name__ == "__main__":
    models = load_models()

    # Executar a função para buscar conversas
    fetch_conversations()

    # Desabilitar treinamento e rodar apenas a interface de comunicação
    interaction_loop_auto(models)
