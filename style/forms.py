from wtforms import Form, StringField, SelectField, IntegerField


class SearchForm(Form):
    #     choices = [('Artist', 'Artist'),
    #                ('Album', 'Album'),
    #                ('Publisher', 'Publisher')]
    #     select = SelectField('Search for music:', choices=choices)
    search = StringField('')

class TrendForm(Form):
    cid = StringField('')
    plot_path = StringField('')
    trend_value = StringField('')
    percent_value = StringField('')
    search = StringField('')


