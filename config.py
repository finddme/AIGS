MODEL_CONFIG= {
  "hidden_size": 768,
  "vocab_size" : 8002,
  "max_len" : 64,
  "epochs": 9999,
  "batch_size": 64,
  "dropout": 0.1,
  "learning_rate": 3e-5,
  "warmup_proportion": 0.1,
  "gradient_accumulation_steps": 1,
  "summary_step": 100,
  "adam_epsilon": 1e-8,
  "warmup_steps": 0,
  "max_grad_norm": 1,
  "logging_steps": 100,
  "evaluate_during_training": True,
  "save_steps": 10,
  "max_steps": -1,
  "threshold": 0.3
}

MONGO_URI = "mongodb://"
DB_NAME = ""
COLLECTION =""
COLLECTION_TEST =""

e_3 = {0:"positive", 1:"negative",2:"neutral"}
