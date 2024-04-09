# from flask_login import UserMixin, login_user, log_out_user, login_required, current_user
from flask import flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
#add database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = '<super secret key >'

# intitialize db
db = SQLAlchemy(app)


#create model
class users(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # unique
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)

    # create string
    def __repr__(self):
        return '<Name %r>' % self.name


# create form class
class userForm(FlaskForm):
    name = StringField('what is your name?', validators=[DataRequired()])
    email = StringField('email', validators=[DataRequired()])
    submit = SubmitField('submit')


class NamerForm(FlaskForm):
    name = StringField('what is your name ', validators=[DataRequired()])
    submit = SubmitField('submit')


@app.route('/users/add', methods=['GET', 'POST'])
def add_users():
    name = None
    form = userForm()
    if form.validate_on_submit():
        user = users.query.filter_by(email=form.email.data).first()
        if user is None:
            user = users(name=form.name.data, email=form.email.data)
            db.session.add(user)
            db.session.commit()
        name = form.name.data
        form.name.data = ''
        form.email.data = ''
        flash('Your account has been created!')
    our_users = users.query.order_by(users.date_added)
    return render_template("add_users.html", form=form,
                           name = name,
                           our_users = our_users)


# name page
@app.route('/name', methods=['GET', 'POST'])
def name():
    name = None
    form = NamerForm()
    if form.validate_on_submit():
        name = form.name.data
        form.name.data = ''
        flash('form submitted successfully')  # message
    return render_template("login.html",
                           name=name,
                           form=form)

# route decorator
# @app.route('/')
# form class
