from pathlib import Path
import glob

def main():
    mypath = str(Path().absolute())
    files_1 = mypath + "/*.xml"
    files = glob.glob(files_1)
    for fichero in files:
        p = fichero.find(".")
        fichero_txt = fichero[:p] + ".txt"
        salida = open(fichero_txt,"w") 
        for l in open(fichero):
            n=l.find("<l n=\"")
            if n>=0:
                inf = l.find("\">") +2
                sup = l.find("</l>")
                salida.write(l[inf:sup])
                salida.write("\n")
        salida.close()






