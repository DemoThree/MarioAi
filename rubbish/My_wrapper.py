class counter:
    def __init__(self,func):
        self.func=func
        self.count=0
    def __call__(self, *args, **kwargs):
        self.count+=1
        print(f"已经调用{self.count}次")
        return self.func( *args, **kwargs)
@counter
def say_hello():
    print("hi")
say_hello()