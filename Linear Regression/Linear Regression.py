import torch
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = x_train*W + b
cost = torch.mean((hypothesis-y_train)**2)
print(cost)

optimizer = optim.SGD([W, b], lr=0.01)
# lr은 학습량

nb_epochs = 3200 # 반복횟수

for epoch in range(nb_epochs +1):
    hypothesis = x_train * W + b #가설 세우고
    cost = torch.mean((hypothesis - y_train) ** 2)  #비용 돌린다

    optimizer.zero_grad() # 계속 누적되는걸 방지

    cost.backward() #계산 : 미분값

    optimizer.step() #학습률(0.01) 을 곱하여 빼줘서 업데이트

    if epoch % 100 == 0: #100단위로
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))


