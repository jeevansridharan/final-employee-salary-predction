def calculate_monthly_salary(annual_salary):
    return annual_salary / 12

def calculate_hourly_rate(annual_salary, hours_per_week=40, weeks_per_year=52):
    return annual_salary / (hours_per_week * weeks_per_year)

def evaluate_model_performance(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Mean Squared Error": mse, "R² Score": r2}