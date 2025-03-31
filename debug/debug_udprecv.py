from theodwyn.networks.comm_prot import ZMQDish
from rohan.common.logging       import Logger
TEST_TOPIC = "test"
with Logger() as logger:
    with ZMQDish( 
        addr        = "udp://192.168.1.48:5560", 
        data_format = "2i",
        timeo       = 1000,
        topic       = TEST_TOPIC,
        #logger      = logger
    ) as dish:
        try: 
            while True:
                one, two =  dish.recv() 
                if two is not None:
                    print(two)
        except KeyboardInterrupt:
            print("... exiting")
