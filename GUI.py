application_name = 'Question Answering from Context'
# pyqt packages
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtGui import QTextCursor, QTextCharFormat
from PyQt5.QtCore import QThread, pyqtSignal

import torch
from torch.nn import functional as F
import tiktoken
import sys

from qt_main import Ui_Application
from gpt import GPT2
from main import compute_device


def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet( 
            parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; \
            min-height: 20px; color:white; background-color: rgb(91, 99, 120); \
            border: 2px solid black; border-radius: 6px;}'
        )
        msg_box.exec()
        
        

# new text generation in multithreading
class inference(QThread):
    update_signal = pyqtSignal(bool)
    parent_class = None
    
    
    def set_param(self, model, tokenizer, context, question, 
        max_new_tokens=128, temperature=1.0, top_k=None):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.question = question
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        
        
    @torch.no_grad()
    def run(self,):
        self.model = self.model.eval()
        
        # Combine question, answer, and prompt template tokens
        prompt = f'Context: {self.context} Question: {self.question} Answer: '
        
        # encode string to list
        input_tok = self.tokenizer.encode_ordinary(prompt) # text encoding
        
        # convert to tensor
        input_tok = torch.tensor(input_tok, dtype=torch.long)
        
        # unsqueeze the batch dimension
        input_tok = input_tok.unsqueeze(0)
        
        # move to compute device
        idx = input_tok.to(compute_device())
        
        # generate
        # idx is (B, T) array of indices in the current context
        for _ in range(self.max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -1024:]
           # forward the model to get the logits for the index in the sequence
            logits, _ = self.model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.temperature
            # optionally crop the logits to only the top k options
            if self.top_k is not None:
                v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next == self.tokenizer._special_tokens['<|endoftext|>']:
                break
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            # decode and update
            decoded = self.tokenizer.decode(idx_next[0].tolist())
            self.parent_class.generated_sentence.append(decoded)
            self.update_signal.emit(True)
            
            



class QT_Action(Ui_Application, QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.model = None
        self.tokenizer = None
        self.generated_sentence = []
        
        # Create the worker thread
        self.inference_thread = inference()
        self.inference_thread.parent_class = self
        
        # load the model
        self.load_model_action()

            
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_clear.clicked.connect(self.clear_action)
        self.toolButton_process.clicked.connect(self.process_action)
        self.inference_thread.update_signal.connect(self.update_action)
    
        
            
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'GPT2':
            num_embed = 768
            num_heads = 12
            num_layers = 12
            self.model = GPT2(num_embed, num_heads, num_layers)
            self.model.apply_lora() 
            
            self.model = self.model.to(compute_device())
            
            # load the pretained weights
            pretrained_path = f'{self.model_name}_finetuned.pth'
            self.model.load_state_dict(torch.load(pretrained_path))
            
            # also load the tokenizer
            self.tokenizer = tiktoken.get_encoding("gpt2")
    
        
    def clear_action(self):
        self.textEdit_context.clear()
        self.textEdit_question.clear()
        self.textEdit_answer.clear()
    
    
    def append_text(self, text):
        cursor = self.textEdit_answer.textCursor()
        format = QTextCharFormat()
        
        # Set the desired color
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text, format)
        
        # Move cursor to the end and set the default color (white) for future text
        cursor.movePosition(QTextCursor.End)
        cursor.setCharFormat(format)
        
        # Ensure the cursor's format is reset
        self.textEdit_answer.setTextCursor(cursor)
        
        
    # process, generate new text
    def process_action(self):
        # model inference
        context = self.textEdit_context.toPlainText()
        question = self.textEdit_question.toPlainText()
        
        # check inputs
        if context == '':
            title = 'Action Error'
            message = 'please enter a context'
            show_message(self, title, message, icon=QMessageBox.Warning)
            return
        elif question == '':
            title = 'Action Error'
            message = 'please enter a question'
            show_message(self, title, message, icon=QMessageBox.Warning)
            return
        
        self.inference_thread.set_param(
            self.model, self.tokenizer, context, question
        )
        
        self.inference_thread.start()
        
        
    def update_action(self, trig):
        text = self.generated_sentence[-1]
        
        # Set generated text to blue
        self.append_text(text)
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()