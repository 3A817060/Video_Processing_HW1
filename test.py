import time
blocks = [1, 2, 3]
cnt = 0
for block in blocks:
    if cnt == 7:
        break
    blocks.insert(1, 0)
    print(block)
    blocks.append(4)  # 這個操作不會影響迴圈
    print(len(blocks))
    time.sleep(0.5)
    cnt += 1 
    