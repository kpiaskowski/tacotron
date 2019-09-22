#paths
data_dir = None
alphabet_path = None
load_checkpoint_path = None

# general
experiment_prefix = 'tacotron'
batch_size = 2
loss_log_step = 5
predict_log_step = 500
save_step = 2000

n_mels = 80
r = 2
max_train_steps = 350
max_eval_steps = 350 // r
iterations = 3000000

# audio
hop_size = 12.5
window_size = 50
sampling_rate = 24000

# encoder
enc_embedding_size = 256
enc_prenet_units = [256, 128]
enc_k = 16
enc_conv_bank_units = 128
enc_conv_proj_units = [128, 128]
enc_highway_units = [128, 128, 128, 128]
enc_gru_units = 128

# decoder
dec_prenet_units = [256, 128]
dec_attention_units = 256
dec_gru_units = [256, 256]

# postprocessor
postproc_k = 8
postproc_conv_bank_units = 128
postproc_conv_proj_units = [256, 80]
postproc_highway_units = [128, 128, 128, 128]
postproc_gru_units = 128
postproc_output_units = 1025
