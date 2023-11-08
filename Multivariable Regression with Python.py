#khaled_elz3balawy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#computeCost
#J(θ) = (1 / (2 * m)) * ∑(i=1 to m) ((hθ(xᵢ) - yᵢ)²)
# ال cost بتتعمل لما ال gdيبعت لها ال theta الجديده كل مره 
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
# Gradient Descent algorithm to update model parameters
# θ_j := θ_j - α * (1 / m) * ∑(i=1 to m) [(hθ(xᵢ) - yᵢ) * xᵢj]

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # عدد المعاملات (الأوزان)
    cost = np.zeros(iters)  # مصفوفة لتخزين تكلفة كل تكرار
    
    for i in range(iters):
        # حساب الفرق بين التنبؤ والقيمة الفعلية
        error = (X * theta.T) - y
        
        for j in range(parameters):
            # حساب الترم الذي يتضمن الفرق والميزة المرتبطة به
            term = np.multiply(error, X[:,j])
            
            # حساب تحديث للمعامل (الوزن) بناءً على متوسط الترم للنقاط البيانات
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        
        # تحديث المعاملات بعد الانتهاء من التكرار
        theta = temp
        
        # حفظ التكلفة (الخطأ) في كل تكرار لمراقبة الأداء
        cost[i] = computeCost(X, y, theta)
    
    return theta, cost


path2 = 'F:\\2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])
#show data
print('data = ')
print(data2.head(10) )
print()
print('data.describe = ')
print(data2.describe())
# rescaling data
data2 = (data2 - data2.mean()) / data2.std()
print()
print('data after normalization = ')
print(data2.head(10) )
# add ones column

data2.insert(0, 'Ones', 1)
# separate X (training data) from y (target variable)

cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]#1,,2,3
y2 = data2.iloc[:,cols-1:cols]#4

print('**************************************')
print('X2 data = \n', X2.head(10))
print('y2 data = \n', y2.head(10))
print('**************************************')
# convert to matrices and initialize theta

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

print('X2 \n', X2)
print('X2.shape = ', X2.shape)
print('**************************************')
print('theta2 \n', theta2)
print('theta2.shape = ', theta2.shape)
print('**************************************')
print('y2 \n', y2)
print('y2.shape = ', y2.shape)
print('**************************************')
# initialize variables for learning rate and iterations

alpha = 0.1
iters = 100
# perform linear regression on the data set

g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
# get the cost (error) of the model

thiscost = computeCost(X2, y2, g2)
print('g2 = ', g2)
print('cost2 = ', cost2[0:50])
print('computeCost = ', thiscost)
print('**************************************')
# get best fit line for Size vs. Price

x = np.linspace(data2.Size.min(), data2.Size.max(), 100)
print('x \n', x)
print('g \n', g2)
f = g2[0, 0] + (g2[0, 1] * x)
print('f \n', f)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Size, data2.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

x = np.linspace(data2.Bedrooms.min(), data2.Bedrooms.max(), 100)
print('x \n', x)
print('g \n', g2)
f = g2[0, 0] + (g2[0, 1] * x)
print('f \n', f)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Bedrooms, data2.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Bedrooms vs. Price')

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Extract the values of Size, Bedrooms, and Price from the data2 DataFrame
x = data2['Size']
y = data2['Bedrooms']
z = data2['Price']

# Plot the data points in 3D
ax.scatter(x, y, z, c='r', marker='o', label='Data Points')

# Set labels for the axes
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

# Set a title for the 3D plot
ax.set_title('3D Scatter Plot of Size, Bedrooms, and Price')



