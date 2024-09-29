#from Metrics import _metrics,feature_explanation
import os
from run import runner
import argparse
import util



def main(args):
   
   util.setup_gpu(args.gpu_id)
   run_=runner(args)

   if(args.training):
      run_.train()
   else:
     run_.test()
     run_.Generate_results()
   print('----done!-------') 



if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   
   parser.add_argument("--Noise_Type", type=str, required=True, help='Noise_Type:CycleGAN,Gaussian,Normal,Pepper')
   parser.add_argument("--Screw_Type", type=str, required=True, help='Screw_Type')
   
   
   parser.add_argument("--epochs", type=int, default=80, help='epochs')
   parser.add_argument("--lr", type=float, default=0.05, help='learning_rate')
   parser.add_argument("--training", type=util.str2bool, default=True, help='model is training')
   parser.add_argument("--gpu_id", type=str, default='', help='gpu card ID')
   parser.add_argument("--image_size", type=int, default=256, help='image_size')
   parser.add_argument("--Noise_Proportion", type=float, default=0.5, help='Proportion of noise if the noise type is Gaussian or salt-and-pepper')
   parser.add_argument("--Anomaly_Network", type=str, default='AE', help='Anomaly Network Backbone')
   parser.add_argument("--noise_filter", type=util.str2bool, default=True, help='noise_filter')

   
   parser.add_argument("--training_folder", type=str, default='trainA', help='training_folder')
   parser.add_argument("--test_folder", type=str, default='', help='test_folder')
   parser.add_argument("--val_folder", type=str, default='valA', help='val_folder')
   parser.add_argument("--recontruction_folder", type=str, default='recontruction', help='recontruction_folder')
   parser.add_argument("--residual_folder", type=str, default='residual', help='residual_folder')
   parser.add_argument("--paired_sample_folder", type=str, default='paired_sample_folder', help='paired_sample_folder')
   parser.add_argument("--model_path", type=str, default='', help='model_path')
   
   
   parser.add_argument("--train_noise_folder", type=str, default='train_noise', help='train_noise_folder')##########
   parser.add_argument("--val_noise_folder", type=str, default='val_noise', help='val_noise_folder')#############





   args = parser.parse_args()
   
   
   if(args.Noise_Type=='Gaussian') or (args.Noise_Type=='pepper'):
      args.model_path=f'models/{args.Anomaly_Network}_{args.Noise_Type}_{str(args.Noise_Proportion)}_{args.Screw_Type}.h5'
   else:
      args.model_path=f'models/{args.Anomaly_Network}_{args.Noise_Type}_{args.Screw_Type}.h5'

   args.training_folder=f'{args.Screw_Type}/{args.training_folder}'
   args.test_folder=f'{args.Screw_Type}/{args.test_folder}'
   args.val_folder=f'{args.Screw_Type}/{args.val_folder}'

   args.train_noise_folder=f'{args.Screw_Type}/{args.train_noise_folder}'####################
   args.val_noise_folder=f'{args.Screw_Type}/{args.val_noise_folder}'##########################


   args.recontruction_folder=f'{args.Screw_Type}/{args.Noise_Type}_{args.recontruction_folder}'
   if not os.path.exists(args.recontruction_folder):
      os.makedirs(args.recontruction_folder)
      
   args.residual_folder=f'{args.Screw_Type}/{args.Noise_Type}_{args.residual_folder}'
   if not os.path.exists(args.residual_folder):
      os.makedirs(args.residual_folder)  
   
   args.paired_sample_folder=f'{args.Screw_Type}/{args.Noise_Type}_{args.paired_sample_folder}'
   if not os.path.exists(args.paired_sample_folder):
      os.makedirs(args.paired_sample_folder) 
      
   print("=" * 80)
   print(f"isTraining:{args.training}")
   print(f"epochs:{args.epochs}")
   print(f"learning_rate:{args.lr}")
   print(f"Screw_Type:{args.Screw_Type}")

   print(f"gpu card:{args.gpu_id}")
   print(f"model path:{args.model_path}")
   print(f"Noise_Type:{args.Noise_Type}")
   print(f"Noise_Proportion:{args.Noise_Proportion}")
   print(f"Anomaly_Network:{args.Anomaly_Network}")
   print(f"image_size:{args.image_size}")
   print(f"noise_filter:{args.noise_filter}")

   print(f"test_folder:{args.test_folder}")
   print(f"recontruction_folder:{args.recontruction_folder}")
   print(f"residual_folder:{args.residual_folder}")
   print(f"training_folder:{args.training_folder}")
   print(f"val_folder:{args.val_folder}")
   print(f"paired_sample_folder:{args.paired_sample_folder}")



   print("=" * 80)
   main(args)
