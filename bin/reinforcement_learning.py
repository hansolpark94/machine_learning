def create_matrix(row, col, val=0):
    """
    Create matrix which has default values of 0
    :param row: int
    :param col: int
    :param val: int
    :return: list with 0
    """
    matrix = []
    for i in range(0, row):
        row = []
        for j in range(0, col):
            row.append(val)
        matrix.append(row)
    return matrix


def get_max_qvalue(present, q_matrix):
    """
    Calculate max q_val for each actions
    :param present: list
    :param q_matrix: matrix
    :return: max_act, max_val
    """
    max = -10000
    max_act = present
    for action in get_next_acts(present):
        q_val = q_matrix[action[0]][action[1]]
        if q_val > max:
            max = q_val
            max_act = action
    return max_act, max


def get_next_acts(present):
    """
    Consider present location, return next possible loc by actions
    :param present: list
    :return: list
    """
    left = [present[0], present[1] - 1]
    right = [present[0], present[1] + 1]
    up = [present[0] + 1, present[1]]
    down = [present[0] - 1, present[1]]

    if left is [1, 1] or left[1] is -1:
        left = present
    if right is [1, 1] or right[1] is 4:
        right = present
    if up is [1, 1] or up[0] is 3:
        up = present
    if down is [1, 1] or down[0] is -1:
        down = present

    return left, right, up, down


def main():
    """
    Grid_world example
    """
    q_matirx = create_matrix(3, 4)
    q_matirx[1][1] = -100
    q_matirx[0][3] = 1
    q_matirx[1][3] = -1
    learning_rate = 0.5
    immediate_reward = -0.04
    discount_factor = 1

    for i in range(0, 100):
        present = [2, 0]
        while present not in [[0, 3], [1, 3]]:
            next, max_qval = get_max_qvalue(present, q_matirx)
            q_matirx[present[0]][present[1]] += learning_rate * (immediate_reward + discount_factor * max_qval
                                                                 - q_matirx[present[0]][present[1]])
            present = next

    for row in q_matirx:
        print(row)


if __name__ == '__main__':
    main()
