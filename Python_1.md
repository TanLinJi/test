

# 其他

- 编译型语言(静态语言)：java

- 解释型语言(脚本语言)：python

- 强制缩进（Python根据缩进来判断代码行和前一个代码行之间的关系）

- python 是一种解释性的编程语言

- IPO程序编写方法：

    ​	input Process output

    输入数据->处理数据->输出数据
    
- 程序无法成功运行时，解释器会提供一个traceback，这是一条记录，指出了解释器尝试运行代码时，在什么地方陷入了困境

- 行长：python文件建议每行都不超过80字符，注释的行长都不超过72字符

- python文件名应使用小写字母和下划线：simple_message.py

- python对缩进的要求很严格 

- pycharm快捷键：

    | Ctrl + /            | 行注释                           |
    | ------------------- | :------------------------------- |
    | Ctrl + Shift + /    | 块注释                           |
    | Ctrl + D            | 复制选定的区域或行到后面或下一行 |
    | shift + alt + 上/下 | 将当前行代码同上下行代码交换     |

## python保留字（35个）

- 查询python中的保留字，严格区分大小写

    ```Python
    import keyword
    print(keyword.kwlist)
    print(len(keyword.kwlist))
    ```


| 关键字   |            |         |          |            |          |
| -------- | ---------- | ------- | -------- | ---------- | -------- |
| `False`  | `None`     | `True`  | `and`    | `as`       | `assert` |
| `async`  | `await`    | `break` | `class`  | `continue` | `def`    |
| `del`    | `elif`     | `else`  | `except` | `finally`  | `for`    |
| `from`   | `global`   | `if`    | `import` | `in`       | `is`     |
| `lambda` | `nonlocal` | `not`   | `or`     | `pass`     | `raise`  |
| `return` | `try`      | `while` | `with`   | `yield`    |          |

## Python中的标识符

- 可以是字母数字下划线，首字符不能是数字
- 不能使用python中的保留字
- 严格区分大小写
- 以下划线开头的标识符一般有特殊意义，应当避免使用相似的标识符
- 允许使用中文作为标识符
- **模块名**尽量短小，并且全部使用小写字母，可以使用下划线分隔多个字母。如：grame_main
- **包名**尽量短小，并且全部使用小写字母，不推荐使用下划线。如：com.ysjpython
- **类名**采用首字母大写形式（Pascal风格）。如：MyClass
- **模块内部的类**采用 “_” + Pascal风格的类名组成，例如阿紫Myclass中的内部类 _InnerMyClass
- **函数、类的属性和方法**的命名全部使用小写字母，多个字母之间使用下划线分隔
- 常量命名时全部采用大写字母，可以使用下划线
- 使用单下划线开头的模块变量或函数是受保护的，在使用 “from ... import ... ”语句从模块中导入时，这些模块变量或函数不能被导入
- 使用双下划线开头的实例变量或方法是类私有的
- 以双下划线开头和结尾的是Python的专用标识，例如：__init\_\_()表示初始化函数

## 注释

- 单行注释： 

    ```python
    # 这是一个单行注释
    ```

- 多行注释：

    ```python
    """
    这是一个多行注释
    这是一个多行注释
    这是一个多行注释
    多行注释本质上是一个字符串
     - 一般对文件、类进行解释说明
    """
    ```

- 中文文档声明注释：

    ```python
    # coding=utf-8
    # 该类注释必须放在文档的第一行代码
    ```
    
- ‘#’ 和注释内容之间空一格

# 第二章

###  2.1 字面量

1. 在代码中写下来的固定的值

2. 包括：数字、字符串、列表、元组、集合、字典

    | 类型               | 描述                   | 说明                                           |
    | ------------------ | ---------------------- | ---------------------------------------------- |
    | 数字（Number）     | 支持：                 |                                                |
    |                    | • 整数（int）          | 如：10、-10                                    |
    |                    | • 浮点数（float）      | 如：13.14、-13.14                              |
    |                    | • 复数（complex）      | 如：4+3j，以`j`结尾表示复数                    |
    |                    | • 布尔（bool）         | True表示真，False表示假；True记作1，False记作0 |
    | 字符串（String）   | 描述文本的一种数据类型 | 由任意数量的字符组成                           |
    | 列表（List）       | 有序的可变序列         | Python中使用最频繁的数据类型，可有序记录数据   |
    | 元组（Tuple）      | 有序的不可变序列       | 可有序记录一堆不可变的Python数据集合           |
    | 集合（Set）        | 无序不重复集合         | 可无序记录一堆不重复的Python数据集合           |
    | 字典（Dictionary） | 无序Key-Value集合      | 可无序记录一堆Key-Value型的Python数据集合      |
    
    ### 说明：
    
    1. 数字类型包含4种子类型，其中布尔值是`int`的子类
    2. 复数类型通过`j`表示虚数部分（如`3j`）
    3. 列表和元组的主要区别是可变性（列表可修改，元组不可修改）
    4. 集合会自动去重，字典通过键值对存储数据

### 2.2 变量

- 格式：变量名称 = 变量值

### 2.3 数据类型

- 使用type()来查看数据的类型

    ```python
    # 使用type()查看数据的类型
    # 字符串类型
    print(type("牛犇"))  # <class 'str'>
    # 整数类型
    int_type = type(666)
    print(int_type)     # <class 'int'>
    # 浮点类型
    num = 12.56
    print(type(num))  	# <class 'float'>
    ```

- 注意：使用type(变量)的方式查看的是数据的类型而不是变量的类型，因为变量无类型

- 初学阶段主要是接触：字符串类型，整数类型和浮点类型

- **字符串类型**：用引号(单引号，双引号)括起来的都是字符串--

    ```Python
    "I'm beautiful!"   # 这样可以灵活的在字符串中包含引号和撇号
    ```

    ```Python
    字符串方法：
    
    name = "ada lovelace"
    print(name.title())  # Ada Lovalace   以首字母大写的方式显示每个单词
    print(name.upper())  # ADA LOVELACE   将字符串改为全部大写
    print(name.lower())  # ada lovelace   将字符串改为全部小写
    
    lanu = "   python    "
    print(lanu.rstrip())   #  删除字符串末尾的空白
    print(lanu.lstrip())   #  删除字符串开头的空白
    print(lanu.strip())	   #  同时删除字符串开头和末尾的空白
    ```

- **字符串拼接：**使用 + 运算符

    ```Python
    first_name = "ada"
    last_name = "lovelace"
    full_name = first_name + " " + last_name
    message = "Hello, " + full_name.title() + "!"
    print("Hello, " + full_name.title() + "!")   # Hello, Ada Lovelace!
    print(message)
    ```

    ```Python
    拼接时，+ 左右两边的操作数必须是字符串类型 
    
    age = 23   # age实际上是一个整型的变量
    pirnt("I am " + age + "year old.")    # 会报错 
    
    print("I am " + str(age) + "year old.")  # 使用str方法先将非字符串值显示成字符串
    
    ```


### 2.4 数据类型转换

- int(x) 将x转换为一个整数

    - 如果将字符串转为数字要求字符串内的内容都是数字
    - 浮点数转为整数会丢失小数部份

- float(x) 将x转换为一个浮点数

- str(x) 将x转换为一个字符串

    - 任何类型都可转为字符串

    ```python
    # 类型转换
    # 转换为int
    num_int = int('11')
    print(type(num_int), num_int)
    
    # 转化为str
    num_str = str(11)
    print(type(num_str), num_str)
    
    # 转换为float
    num_float = str("12.65")
    print(type(num_float), num_float)
    ```

### 2.5 标识符和保留字

python中的保留字：

```python
and as assert break class continue def del elif else except finally for from False global if import in is lambda nonlocal not None or pass raise return try True while with yield await async
```

```python
# 在python中查询
import keyword
print(keyword.kwlist)
print('关键词个数为：', len(keyword.kwlist), sep='')
```

- 保留字是严格区分大小写的

python中的标识符就是变量名，注意命名规范，**区分大小写**

- 标识符**只能使用**：字母、数字、下划线、中文，但是不能使用数字开头
- 标识符**不能使用保留字**
- **模块名**尽量短小，且全部使用小写字母，单词之间用下划线分隔
- **包名**尽量短小，且全部使用小写字母，不推荐使用下划线，可以是点：com.yspt
- **类名**采用驼峰命名法，即每个单词的首字母都大写，不使用下划线
- **类的实例**采用小写，在单词之间添加下划线
- **模块内部的**类在满足类的命名规则下加前缀下划线：_MyClass
- **函数、类的属性、方法**全部使用小写字母，多个字母之间用下划线分隔
- **常量**命名全部大写，可以使用下划线
- 使用单下划线开头的模块变量或函数是受保护的，使用‘form xx import *’语句导入时，这些模块变量或函数时不能被导入的
- 使用双下划线“__”开头的实例变量或方法是类私有的
- 以**双下划线开头和结尾**是Pyhon的专用标识：__init()__
- 在类中，使用一个空行来分隔方法
- 在模块中，使用两个空行来分隔类
- 对于每个类，都应在其后紧跟一个文档字符串说明此类的用途
- 对于每个模块，都应包含一个文档字符串来说明这个模块中的类的大体作用
- 需要同时导入标准库中的模块和自己的模块时，应该先导入标准库中的模块，再添加一个空行，再导入自己的模块



### 2.6 运算符

以下是 Python 的主要运算符列表：

| 运算符   | 描述                     | 例子                                   |
| -------- | ------------------------ | -------------------------------------- |
| `+`      | 加法                     | `5 + 3 → 8`                            |
| `-`      | 减法                     | `5 - 3 → 2`                            |
| `*`      | 乘法                     | `5 * 3 → 15`                           |
| `/`      | 除法                     | `10 / 2 → 5.0`                         |
| `//`     | 整除                     | `10 // 3 → 3`                          |
| `%`      | 取模                     | `10 % 3 → 1`                           |
| `**`     | 幂运算                   | `2 ** 3 → 8`                           |
| `=`      | 赋值                     | `x = 5`                                |
| `+=`     | 加后赋值                 | `x += 3`（等价于 `x = x + 3`）         |
| `-=`     | 减后赋值                 | `x -= 2`（等价于 `x = x - 2`）         |
| `*=`     | 乘后赋值                 | `x *= 4`（等价于 `x = x * 4`）         |
| `/=`     | 除后赋值                 | `x /= 2`（等价于 `x = x / 2`）         |
| `//=`    | 整除赋值                 | `x //= 3`（等价于 `x = x // 3`）       |
| `%=`     | 取模赋值                 | `x %= 3`（等价于 `x = x % 3`）         |
| `**=`    | 幂运算赋值               | `x **= 2`（等价于 `x = x ** 2`）       |
| `&=`     | 按位与赋值               | `x &= 3`（等价于 `x = x & 3`）         |
| `|=`     | 按位或赋值               | `x |= 3`（等价于 `x = x | 3`）         |
| `^=`     | 按位异或赋值             | `x ^= 3`（等价于 `x = x ^ 3`）         |
| `<<=`    | 左移赋值                 | `x <<= 1`（等价于 `x = x << 1`）       |
| `>>=`    | 右移赋值                 | `x >>= 1`（等价于 `x = x >> 1`）       |
| `==`     | 等于                     | `5 == 5 → True`                        |
| `!=`     | 不等于                   | `5 != 3 → True`                        |
| `>`      | 大于                     | `5 > 3 → True`                         |
| `<`      | 小于                     | `5 < 3 → False`                        |
| `>=`     | 大于等于                 | `5 >= 5 → True`                        |
| `<=`     | 小于等于                 | `5 <= 3 → False`                       |
| `and`    | 逻辑与                   | `(5>3) and (2<4) → True`               |
| `or`     | 逻辑或                   | `(5<3) or (2<4) → True`                |
| `not`    | 逻辑非                   | `not (5<3) → True`                     |
| `&`      | 按位与                   | `5 & 3 → 1`                            |
| `|`      | 按位或                   | `5 | 3 → 7`                            |
| `^`      | 按位异或                 | `5 ^ 3 → 6`                            |
| `~`      | 按位取反                 | `~5 → -6`                              |
| `<<`     | 左移                     | `5 << 1 → 10`                          |
| `>>`     | 右移                     | `5 >> 1 → 2`                           |
| `in`     | 成员存在检查             | `3 in [1,2,3] → True`                  |
| `not in` | 成员不存在检查           | `4 not in [1,2,3] → True`              |
| `is`     | 对象身份相同             | `a is b`（当 `a` 和 `b` 是同一对象时） |
| `is not` | 对象身份不同             | `a is not b`                           |
| `:=`     | 海象运算符（赋值表达式） | `if (n := len(a)) > 5:`                |

 **说明：**

1. **位运算符**操作的是整数的二进制形式（如 `5` 的二进制为 `101`）。
2. **身份运算符**（`is`/`is not`）比较对象的内存地址，而 `==` 比较值。
3. **海象运算符**（Python 3.8+）允许在表达式中赋值（如条件判断、列表推导式等）。



### 2.7 字符串扩展

- 字符串有三种定义方法：

    ```python
    # 单引号
    name = '牛犇'
    # 双引号
    name = "牛犇"
    # 三引号(接收了是变量，没接收就是注释)
    name = """牛犇"""
    ```

- 如果字符串的内容本身就有单引号或者多引号：

    1. 单引号定义法：可以包含双引号
    2. 双引号定义发：可以包含单引号
    3. 使用转义字符‘\’来解除效用，变成普通字符串

    ```python
    name = '牛犇"'
    name = "牛犇'"
    name = "牛犇\'"
    ```

- 字符串拼接

    1. 字符串和字符串拼接，用 +

        ```python
        name = "牛犇" + "的你"
        ```

    2. 字符串无法和其他类型数据通过“+”进行拼接

    

- 字符串格式化

    - 多个变量占位，变量要用括号括起来，并按照占位的顺序填入
    - 使用小写的s，不能使用大写的s

    ```python
    name = "张三"
    message = "姓名：%s" % name
    print(message)
    
    name = "张三"
    salary = 12569
    message = "姓名：%s, 工资：%s" % (name, salary)
    print(message)
    ```

    - 占位符有：%s, %d ,%f 

# 第三章 列表

### 3.1 列表基础操作

- 列表是由一系列按特定顺序排列的元素组成，是有序集合，列表的元素之间没有任何关系，通常列表名称是一个表示复数的名字

    ```Python
    names = ['john', 'Alan', 'Bob']  #  列表由方括号括起来
    print(names)   # 输出['john', 'Alan', 'Bob']
    ```

- 访问列表元素(python中下标从 0 而不是从 1 开始)

    ```Python
    names = ['john', 'Alan', 'Bob'] 
    print(names[0])   # john  只打印该元素，不包含方括号和引号
    print(names[0].title())
    print(names[-1])  # 使用 -1 访问列表的最后一个元素
    ```

- 可以使用索引 -1 访问最后一个元素，-2 访问倒数第二个元素等等

- 判断列表是否为空：

    ```Python
    cars = []
    if cars:
        print("Welcome!")
    else:
        print("We have no car!")
    ```

- 修改列表元素

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    motors[0] = 'ducati'
    print(motors)   #  ['ducati', 'yamaha', 'suzuki']
    ```

- 在列表末尾添加元素 append()

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    motors.append('ducati')
print(motors)  #  ['honda', 'yamaha', 'suzuki', 'ducati']
    
    可以使用方法 append 动态的创建列表（这种方式非常常用）
    motors = []
    motors.append('honda')
    motors.append('yamaha')
    motors.append('suzuki')
    ```
    
- 使用 insert() 方法在列表中指定位置插入元素 

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    motors.insert(0,'ducati')   #  在索引值为 0 的位置添加新元素
    print(motors)   #  ['ducati', 'honda', 'yamaha', 'suzuki']
    ```

- 使用 del 语句删除列表中元素（删除之后无法再访问这个被删除的元素）

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    del motors[0]    #  删除索引值为 0 处的元素
    print(motors)  #  ['yamaha', 'suzuki']
    ```

- 使用 pop() 方法删除列表末尾的元素

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    my_motor = motors.pop()
    print(motors)     #   ['honda', 'yamaha']
    print(my_motor)   #   suzuki
    ```

- 使用 pop() 方法来删除列表中指定位置的元素

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    my_motor = motors.pop(0)  #  弹出索引位置为 0 处的元素
    print(motors)     #  ['yamaha', 'suzuki']
    print(my_motor)   #  honda
    ```

- 关于 pop() 方法和 del 语句，这两个都会从列表中删除一个元素，如果这个元素以后不再使用就用 del 语句，如果要使用就用 pop 方法

- 使用方法 remove() 根据值删除列表中的元素

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    motors.remove('honda')   # 如果试图删除一个列表中不存在的元素则会报错
    print(motors)   #  ['yamaha', 'suzuki'] 	
    ```

    注意：remove() 方法只删除列表中第一个符合条件的值，如果存在多个，应该使用循环

    可以把要删除的元素先存储在变量中

    ```Python
    motors = ['honda', 'yamaha', 'suzuki']
    too_expensive = 'honda'
    motors.remove(too_expensive)
    print(motors)   #  ['yamaha', 'suzuki'] 
    ```

### 3.2 列表排序

- 使用 sort() 方法对列表进行永久性排序

    ```Python
    cars = ['bmw','audi','toyota','subaru']
    cars.sort()   #  对cars列表按字母进行永久性排序(升序)
    print(cars)   #  ['audi', 'bmw', 'subaru', 'toyota']
    cars.sort(reverse=True)   #  对cars列表按字母进行永久性排序(升序)
    print(cars)   #  ['toyota', 'subaru', 'bmw', 'audi']
    ```

- 使用 sorted() 函数对列表进行临时性排序

    ```Python
    cars = ['bmw','audi','toyota','subaru']
    print(sorted(cars))   #  ['audi', 'bmw', 'subaru', 'toyota']   使用 sorted() 函数后，列表本来的顺序并没有改变
    print(sorted(cars,reverse=True))   #  ['toyota', 'subaru', 'bmw', 'audi']
    print(cars)   #  ['audi', 'bmw', 'subaru', 'toyota']
    ```

- 接收一个 使用 sorted() 函数对列表进行临时性排序后的列表

    ```
    cars = ['bmw','audi','toyota','subaru']
    sorted_cars = sorted(cars)
    print(sorted_cars)   #  ['audi', 'bmw', 'subaru', 'toyota']
    ```

- 使用方法 reverse() 永久性反转列表的顺序

    ```Python
    cars = ['bmw','audi','toyota','subaru']
    cars.reverse()
    print(cars)  #  ['subaru', 'toyota', 'audi', 'bmw']
    ```

- 使用 len() 函数确定列表的长度

    ```Python
    cars = ['bmw','audi','toyota','subaru']
    car_num = len(cars)
    print(car_num)   #  4
    ```


### 3.3 操作列表

- 使用 for 循环打印列表元素

    ```Python
    cars = ['bmw','audi','toyota','subaru']
    for car in cars:
    	print(car)
    ```

- for循环中，靠的是缩进来区分内部或外部的语句

    ```Python
    cars = ['bmw','audi','toyota','subaru']
    for car in cars:
    	print("I have a car: " + car)
    	print("That's beautiful!\n")
    ```

    在for循环后，没有缩进的代码都只执行一次


### 3.4 数字列表

- 使用 range 函数创建一系列数字

    ```Python
    for value in range(1,5):  #  是左闭右开区间
    	print(value)  # 1 2 3 4
    ```

- 使用 list() 函数将 range() 的结果直接转换成列表

    ```Python
    numbers = list(range(1,6)) 
    print(numbers)  #  [1, 2, 3, 4, 5]
    ```

- 使用 range() 函数指定步长

    ```Python
    numbers = list(range(1,12,2))  # 从 1 开始，以 2 为步长，不断增加到 12
    print(numbers)  #  [1, 3, 5, 7, 9, 11]
    ```

- 将1~10的平方放在一个列表中

    ```Python
    squares = []
    for value in range(1,11):
    	squares.append(value**2)
    print(squares)  #  [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    ```

- 对数字列表进行简单的统计计算

    ```Python
    digits = [1,2,3,4,5,6,7,8,9,0]
    min(digits)  # 0
    max(digits)  # 9
    sum(digits)  # 45
    ```

- 列表解析：

    ```Python
    squares = [value**2 for value in range(1,11)]
     1. squares是列表名
     2. value**2 是表达式
     3. 最后是循环语句，用于给表达式提供值
        
    等价于上边的平方列表
    ```


### 3.5 使用列表的一部分

- 切片

    ```Python
    players = ['charles','martina','michael','florence','eli']
    print(players[0:3])  # ['charles', 'martina', 'michael']  打印列表的一部分（一个切片）
    print(players[:2]) # 没有指定第一个索引，将自动从列表头开始    ['charles', 'martina']
    print(players[2:]) # 没有治党第二个索引，将自动提取到列表末尾  ['michael', 'florence', 'eli']
    ```

- 输出名单最后三名队员

    ```Python
    players = ['charles','martina','michael','florence','eli']
    print(players[-3:])  #  ['michael', 'florence', 'eli']
    
    ```

- 遍历切片

    ```Python
    players = ['charles','martina','michael','florence','eli']
    for player in players[:3]:
        print(player.title())   #  遍历前三名队员
    ```

- 复制列表

    ```Python
    foods = ['pizza','falafel','carrot cake']
    new_foods = foods[:]  # 同时省略起始索引和终止索引来创建一个包含整个列表的切片
    
    foods 和 new_foods 的内容相同，但是地址不同，对 new_foods 中的元素进行修改不会影响原来的 foods 中的元素
    
    new_foods = foods  # 这种方式会使 new_foods 关联到 foods 包含的列表，对其中任意一个的修改都会影响对方
    ```

# 第四章 元组

- 列表是可以被修改的，元组的值是不能被修改的

- Python 将不能修改的值称为不可变的，而不可变的 列表 被称为元组

- 定义元组

    ```Python
    dimensions = (100, 200)
    print(dimensions[0])   # 像访问列表元素一样访问元组元素
    print(dimensions[1])
    
    dimensions[0] = 20   # 会报错，Python 不运行修改元组元素
    ```

- 遍历元组

    ```Python
    dimensions = (100, 200)
    
    for dim in dimensions:
        print(dim)
    ```

- 修改元组变量

  ```Python
  dimensions = (100, 200)
  dimensions = (200, 400)   # 定义了一个新的元组存储到dimensions中，因为给元组变量赋值是可以的
  ```
  
  

# 第五章 条件语句

- if语句：

    ```Python
    cars = ['audi','bmw','subaru','toyota']
    for car in cars:
        if car == 'bmw':   #  == 运算符，值相等时返回True,否则返回False
            print(car.upper())
        elif car == 'subaru':
            print(car.lower())
        else:
            print(car.title())
    
    age = 12
    if age < 4:
        price = 0
    elif age < 18:
        price = 10
    else:
        price = 20 
    print("The price is " , price)  # price 看似是局部变量，但是对python来说都可以使用
    ```

- 如果想要大小写没这么重要，可以全部转换成小写，再判断

    ```Python
    car.lower() == 'bmw
    判断两个不相等 用 !=
    
    if car != 'daz':
        print(car.upper())
    ```

- 条件语句中可以包含各种数学比较： > , < , >= , <= , == , !=

- 使用 and 检查多个条件

    ```Python
    my_age = 23
    his_age = 27
    if my_age <= 24 and his_age >=26:  # 可以添加括号 if (my_age <= 24) and (his_age >=26):
    	print('haha')
    ```

- 使用 or 检查多个条件

    ```Python
    my_age = 15
    his_age = 27
    if my_age <= 10 or his_age >=26:  # 可以添加括号 if (my_age <= 10) or (his_age >=26):
    	print('haha')
    ```

- 使用 in 关键字检查特定值是否包含在列表中

    ```Python
    cars = ['audi','bmw','subaru','toyota']
    if 'bmw' in cars:
        print("yes!")
    else:
        print("No!")
    ```

- 使用关键字 not in 检查特定值是否不包含在列表中

    ```Python
    cars = ['audi','bmw','subaru','toyota']
    if 'kasa' not in cars:
        print("yes!")
    else:
        print("No!")
    ```

    