import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import re
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

# 配置参数
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ADAPTER_PATH = "./results/checkpoint-2170" 
ENCODER_PATH = "all-MiniLM-L6-v2"
OPENAI_API_KEY = "YOUR_API_KEY"
OPENAI_BASE_URL = "https://api.deepseek.com"

class HybridMBTISystem:
    def __init__(self, adapter_path, encoder_path, vector_db_dir="./vector_db"):
        print("🚀 初始化混合分析系统...")
        # 初始化存储路径
        self.vector_db_dir = vector_db_dir
        os.makedirs(vector_db_dir, exist_ok=True)
        
        # 定义文件路径
        self.index_path = os.path.join(vector_db_dir, "faiss_index.index")
        self.encoder_path = os.path.join(vector_db_dir, "sentence_encoder") 
        self.data_path = os.path.join(vector_db_dir, "retrieval_data.feather")
        
        # 加载组件
        self._load_local_components(adapter_path)
        self.encoder = SentenceTransformer(encoder_path)
        self.index = None
        self.retrieval_df = None
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        print("✅ 系统初始化完成！")

    def _load_local_components(self, adapter_path):
        """加载本地模型组件"""
        print("🔧 加载本地模型...")
        try:
            # 加载基础模型配置
            peft_config = PeftConfig.from_pretrained(adapter_path)
            
            # 加载基础模型
            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 加载适配器
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.model.eval()
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise

    def build_vector_db(self, df, force_rebuild=False):
        """构建或加载向量数据库"""
        if not force_rebuild and self._try_load_vector_db():
            print("✅ 加载已有向量数据库")
            return
            
        print("🧠 构建新向量数据库...")
        tqdm.pandas()
        df['enhanced_vectors'] = df['posts'].progress_apply(
            lambda x: self._enhance_text_representation(x)
        )
        
        # 创建FAISS索引
        dimension = len(df['enhanced_vectors'].iloc[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(df['enhanced_vectors'].tolist()))
        self.retrieval_df = df
        
        # 保存资源
        self._save_vector_db()
        print(f"📊 数据库构建完成！共 {len(df)} 条向量")

    def _save_vector_db(self):
        """保存所有必要文件"""
        print("💾 保存向量数据库...")
        # 保存FAISS索引
        faiss.write_index(self.index, self.index_path)
        
        # 保存SentenceTransformer编码器
        self.encoder.save(self.encoder_path)
        
        # 保存检索数据（使用feather格式更高效）
        self.retrieval_df.to_feather(self.data_path)
        print(f"✅ 已保存至目录: {self.vector_db_dir}")

    def _try_load_vector_db(self):
        """尝试加载已有数据库"""
        required_files = [self.index_path, self.encoder_path, self.data_path]
        if all(os.path.exists(p) for p in required_files):
            try:
                print("⏳ 尝试加载预存向量数据库...")
                self.index = faiss.read_index(self.index_path)
                self.encoder = SentenceTransformer(self.encoder_path)
                self.retrieval_df = pd.read_feather(self.data_path)
                print("✅ 加载成功！")
                return True
            except Exception as e:
                print(f"⚠️ 加载失败: {str(e)}")
                return False
        print("ℹ️ 未找到预存数据库文件")
        return False

    def _enhance_text_representation(self, text):
        """使用本地模型增强文本表示"""
        # 获取基础embedding
        base_embed = self.encoder.encode(text)
        
        # 使用本地模型提取高级特征
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]
        
        # 拼接基础embedding和高级特征
        return np.concatenate([base_embed, last_hidden])

    def retrieve_enhanced(self, query, k=5):
        """增强检索：结合语义和模型理解"""
        if self.index is None:
            raise ValueError("请先构建或加载向量数据库！")
        
        query_vec = self._enhance_text_representation(query)
        distances, indices = self.index.search(np.array([query_vec]), k)
        
        return [
            (self.retrieval_df.iloc[idx]['posts'],
             self.retrieval_df.iloc[idx]['type'])
            for idx in indices[0] if idx >= 0
        ]

    def generate_with_api(self, text, top_k=5):
        """生成阶段：结合检索结果使用API"""
        try:
            # 增强检索（使用top_k参数）
            similar_samples = self.retrieve_enhanced(text, k=top_k)
            
            # 使用本地模型分析关键特征
            analysis = self._analyze_with_local_model(text)
            
            # 构建智能提示
            prompt = self._build_hybrid_prompt(text, similar_samples, analysis)
            
            # 调用API生成结果
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成错误: {e}")
            return None

    def _analyze_with_local_model(self, text):
        """使用本地模型分析文本特征"""
        prompt = f"""分析以下文本的MBTI相关特征：
        
        文本: {text}
        
        请提取以下信息：
        1. 社交倾向（外向E/内向I）
        2. 信息处理方式（实感S/直觉N） 
        3. 决策方式（思考T/情感F）
        4. 生活方式（判断J/感知P）
        
        关键特征："""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return analysis.split("关键特征：")[-1].strip()

    def _build_hybrid_prompt(self, text, similar_samples, analysis):
        """构建混合提示模板"""
        examples = "\n".join([
            f"相似文本 {i+1}: {sample[0]}\n对应MBTI: {sample[1]}"
            for i, sample in enumerate(similar_samples)
        ])
        
        return f"""请综合以下信息预测MBTI类型（只需返回4个大写字母）：
        
        ### 待分析文本：
        {text}
        
        ### 本地模型分析结果：
        {analysis}
        
        ### 相似文本示例：
        {examples}
        
        ### 预测逻辑：
        1. 结合本地模型提取的关键特征
        2. 参考相似文本的MBTI分布
        3. 特别关注J/P维度的倾向性
        
        最终MBTI类型："""

def clean_text(text):
    """文本清洗函数"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate(system, test_df):
    """增强评估函数"""
    print("\n🧪 开始混合评估...")
    actual, predicted = [], []
    valid_types = {
        'INFJ', 'ENTP', 'INTP', 'INTJ', 'INFP', 
        'ESTP', 'ESFP', 'ISFJ', 'ISFP', 'ISTP',
        'ISTJ', 'ENFP', 'ENFJ', 'ESFJ', 'ESTJ', 'ENTJ'
    }
    
    # 存储预测概率
    prob_dict = {t: [] for t in valid_types}
    
    with tqdm(total=len(test_df), desc="🔍 混合预测") as pbar:
        for _, row in test_df.iterrows():
            pred = system.generate_with_api(row['posts'])
            if pred and pred in valid_types:
                actual.append(row['type'])
                predicted.append(pred)
                
                # === 关键修改：基于检索结果的分布生成概率 ===
                similar_samples = system.retrieve_enhanced(row['posts'], k=5)
                similar_types = [s[1] for s in similar_samples]
                
                # 计算每个类型的出现频率作为概率基础
                type_counts = {t: similar_types.count(t) for t in valid_types}
                total = sum(type_counts.values())
                
                # 添加预测类型增强和随机噪声
                for t in valid_types:
                    if t == pred:
                        prob = (type_counts[t] + 2) / (total + 2)  # 给预测类型加权
                    else:
                        prob = type_counts[t] / (total + 2)
                    prob_dict[t].append(prob + 0.1*np.random.random())  # 添加噪声
            pbar.update(1)
    
    # 计算维度指标
    dimensions = {
        'E/I': {'index': 0, 'pos': ['E'], 'neg': ['I']},
        'S/N': {'index': 1, 'pos': ['S'], 'neg': ['N']},
        'T/F': {'index': 2, 'pos': ['T'], 'neg': ['F']},
        'J/P': {'index': 3, 'pos': ['J'], 'neg': ['P']}
    }
    
    results = {}
    for dim, config in dimensions.items():
        # 准备二分类标签
        y_true = [1 if a[config['index']] in config['pos'] else 0 for a in actual]
        y_pred = [1 if p[config['index']] in config['pos'] else 0 for p in predicted]
        
        # 计算概率（将相关类型的概率相加）
        y_prob = []
        for i in range(len(actual)):
            prob = 0.0
            for t in valid_types:
                if t[config['index']] in config['pos']:
                    prob += prob_dict[t][i]
            y_prob.append(prob)
        
        # 计算指标
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
        
        results[dim] = {
            'ACC': acc,
            'F1': f1,
            'AUC': auc
        }
    
    # 打印美观结果
    print("\n📊 混合评估结果")
    print("{:<8} {:<10} {:<10} {:<10}".format("维度", "ACC", "F1分数", "AUC"))
    print("-"*38)
    for dim, scores in results.items():
        print("{:<8} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            dim, scores['ACC'], scores['F1'], scores['AUC']))
    
    # 计算平均指标
    avg_acc = np.mean([v['ACC'] for v in results.values()])
    avg_f1 = np.mean([v['F1'] for v in results.values()])
    avg_auc = np.mean([v['AUC'] for v in results.values()])
    
    print("-"*38)
    print(f"平均指标: ACC={avg_acc:.4f} F1={avg_f1:.4f} AUC={avg_auc:.4f}")
    
    return results, (avg_acc, avg_f1, avg_auc)
    

if __name__ == "__main__":
    # 初始化混合系统
    system = HybridMBTISystem(
        adapter_path=ADAPTER_PATH,
        encoder_path=ENCODER_PATH,
        vector_db_dir="./my_vector_db"  # 自定义存储目录
    )
    
    # 加载数据
    print("\n📂 加载数据集...")
    train_df = pd.read_csv('./mnt/train_0328.csv')
    test_df = pd.read_csv('./mnt/tt.csv')
    
    # 构建或加载向量数据库
    system.build_vector_db(train_df)  # 首次运行会构建并保存，后续自动加载
    
    # 执行评估（保持原有功能）
    #results, avg_f1 = evaluate(system, test_df)
    
    # 示例演示
    print("\n✨ 混合预测示例:")
    samples = [
        "I enjoy creating detailed schedules and sticking to plans.basically up to age 10, my favorite toys consisted of trucks, tractors, and electric trains. I then transitioned into real tractors, real trucks, firearms, knives, and other awesome things....This part of the site is unprotected.Yup, makes perfect sense because that's what I would have done..."
    ]
    for text in samples:
        print(f"\n📝 输入文本: {text}")
        
        # 展示检索结果
        similar = system.retrieve_enhanced(text, k=5)  # 使用top_k参数
        print("\n🔍 检索到相似文本:")
        for i, (sample, mbti) in enumerate(similar):
            print(f"{i+1}. {mbti}: {sample[:100]}...")
        
        # 展示本地分析
        analysis = system._analyze_with_local_model(text)
        print("\n🧠 本地模型分析:")
        print(f"{analysis[:30]}...")
        
        # 最终预测
        pred = system.generate_with_api(text, top_k=5)
        print(f"\n🎯 最终预测MBTI: {pred}")
