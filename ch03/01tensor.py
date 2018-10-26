#텐서플로를 라이브러리 임포트
import tensorflow as tf

#상수를 hello변수에 저장
hello = tf.constant('Hello, TensorFlow!')
print(hello)

#연산
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)
print(c)

#실제 연산 실행
sess = tf.Session()

print(sess.run(hello))
print(sess.run([a,b,c]))

#플레이스 홀더 = 매개변수
X = tf.placeholder(tf.float32, [None, 3])
print(X)

#X에 넣을 자료 정의
x_data = [[1,2,3],[4,5,6]]

#정규분포로 랜덤하게 변수를 설정
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

#수식
expr = tf.matmul(X, W) + b

#연산 실행
sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict = {X: x_data}))

sess.close
