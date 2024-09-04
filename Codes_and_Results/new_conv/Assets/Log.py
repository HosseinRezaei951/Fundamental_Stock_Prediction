import os

class Log:
    def __init__(self, outPut_path, fileName):
        self.oPath = outPut_path
        self.fName = fileName
        self.log_string = ""

    def __call__(self, newLine_log, print_tag = True):
        log_str = "\n"
        log_str += f'{newLine_log}'

        if (print_tag == True):
            print(log_str)

        self.log_string += log_str  

    def write_log(self):
        log_file = f'{self.oPath}/{self.fName}.txt'
        f = None
        if not os.path.exists(log_file):
            f = open(log_file, "w")
            f.write(self.log_string)
            f.close()
        else:
            with open(log_file, 'a') as f:
                f.write(self.log_string)
                f.close()
    