from time import now, sleep


fn cpu_bound_task() -> None:
    var sum: Int64 = 0
    sleep(1)
    # for i in range(1000000000):

    var i:Int64=0

    while i <= 1000000000:
        sum +=i*i
        i=+i+1



fn main() -> None:
    var start:Int = now()

    cpu_bound_task()

    var end:Int=now()

    var sub:Float64=end-start
    sub=sub/pow(10,9)-1.0

    print("Mojo elapsed time:"+ str(sub) +" seconds")
