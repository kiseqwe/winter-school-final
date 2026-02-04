import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Вимикаємо GPU (бо ми на CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    print("⚙️ Loading model for inference on CPU...")
    
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Визначаємо шлях до адаптерів відносно скрипта
    # (шукаємо папку taxi_dpo_cpu_final поруч із цим файлом)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ADAPTER_PATH = os.path.join(script_dir, "taxi_dpo_cpu_final")

    # 1. Завантажуємо БАЗОВУ модель
    print(f"⏳ Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    # 2. Надягаємо на неї твої АДАПТЕРИ
    print(f"🔗 Loading adapters from {ADAPTER_PATH}...")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("✅ Adapters loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading adapters: {e}")
        print("Спершу запусти train_dpo.py, щоб створити модель!")
        return

    # 3. Тестовий промпт
    prompt_text = "Я планую поїздку в NYC. День: Friday, час: 18:00. Пасажирів: 2. Скільки це займе часу?"
    
    # Формат ChatML
    messages = [{"role": "user", "content": prompt_text}]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to("cpu")

    print("🤖 Generating response... (Please wait)")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.95
        )

    # Декодуємо відповідь
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*30)
    print("🗣️ MODEL RESPONSE:")
    print("="*30)
    # Показуємо тільки відповідь асистента
    if "assistant" in response:
        print(response.split("assistant")[-1].strip())
    else:
        print(response)
    print("="*30)

if __name__ == "__main__":
    main()