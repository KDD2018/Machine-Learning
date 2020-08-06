from flask import Flask


app = Flask(__name__)

@app.route('/')
def hello_world():

   return 'hool'

@app.route('/name')
def hello_name():

   return 'laowang'

@app.route('/age')
def hello_age():

   return '18'

if __name__ == '__main__':
   app.run(port=9988)