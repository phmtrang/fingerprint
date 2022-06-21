import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import predict

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('Không có ảnh nào được chọn')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Ảnh được upload thành công và hiển thị kết quả')
		fileResult = predict.similar(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		# print (fileResult)
		return render_template('upload.html', filenames=['uploads/'+filename,'validate/'+ fileResult])
	else:
		flash('Chỉ cho phép các kiểu file có đuôi là -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<directory>/<filename>')
def display_image(directory,filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static',filename= directory+'/'+filename), code=301)

if __name__ == "__main__":
    app.run()