import numpy as np
import cvxpy as cp
import json

ACTION_UP = 'UP'
ACTION_LEFT = 'LEFT'
ACTION_DOWN = 'DOWN'
ACTION_RIGHT = 'RIGHT'
ACTION_STAY = 'STAY'
ACTION_SHOOT = 'SHOOT'
ACTION_HIT = 'HIT'
ACTION_CRAFT = 'CRAFT'
ACTION_GATHER = 'GATHER'
ACTION_NONE = 'NONE'

STATE_DORMANT = 'D'
STATE_READY = 'R'

POSITION_W = 'W'
POSITION_N = 'N'
POSITION_E = 'E'
POSITION_S = 'S'
POSITION_C = 'C'

ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_STAY, ACTION_SHOOT, ACTION_HIT, ACTION_CRAFT, ACTION_GATHER, ACTION_NONE]

POSITIONS = [POSITION_W , POSITION_N, POSITION_E, POSITION_S, POSITION_C]
MM_STATES = [STATE_DORMANT, STATE_READY]
ARROWS = [0,1,2,3]
MATS = [0,1,2]
HEALTHS = [0,25,50,75,100]


# OPERATION RESULTS
GO_LEFT = {
    'C' : 'W',
    'E' : 'C',
}

GO_STAY = {
    'C':'C',
    'W':'W',
    'E':'E',
    'S':'S',
    'N':'N',
}

GO_RIGHT = {
    'W':'C',
    'C':'E'
}

GO_UP ={
    'C':'N',
    'S':'C'
}

GO_DOWN = {
    'C':'S',
    'N':'C'
}


MOVEMENT_PROB= {
    'C':0.85,
    'E':1,
    'W':1,
    'N':0.85,
    'S':0.85
}

HIT_PROB_SUCCESS = {
    'C':0.1,
    'E':0.2,
}


GO_UNSUCCESS = {
    'C':'E',
    'N':'E',
    'S':'E',
    ### will never happen
    'W':'W',
    'E':'E',
}

SHOOT_PROB_SUCCESS = {
    'C':0.5,
    'W':0.25,
    'E':0.9,
}

step_cost = -5

def isValidAction(state, action):
    position, mat, arrow, mm_state, health = state

    if health == 0:
        return action == ACTION_NONE

    if action == ACTION_NONE:
        return health == 0

    if action == ACTION_STAY:
        return True

    if action == ACTION_CRAFT:
        return position == POSITION_N and mat > 0

    if action == ACTION_GATHER:
        return position == POSITION_S

    if action == ACTION_SHOOT:
        return (position == POSITION_C or position == POSITION_E or position == POSITION_W) and arrow > 0

    if action == ACTION_HIT:
        return (position == POSITION_C or position == POSITION_E)

    if action == ACTION_UP:
        return (position == POSITION_S or position == POSITION_C)

    if action == ACTION_DOWN:
        return (position == POSITION_N or position == POSITION_C)
    
    if action == ACTION_LEFT:
        return (position == POSITION_E or position == POSITION_C)
    
    if action == ACTION_RIGHT:
        return (position == POSITION_W or position == POSITION_C)



# a state and a valid action needs to be passed
def transformation_lp(state, action):
    position, mat, arrow, mm_state, health = state


    probability=[]
    new_states=[]
    rewards = []


    if(action == ACTION_NONE):
        return [1], [state], [-step_cost]
        

    elif(action == ACTION_LEFT):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_LEFT[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)


    elif(action == ACTION_DOWN):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_DOWN[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])
    
        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)

    elif(action == ACTION_RIGHT):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_RIGHT[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)

    elif(action == ACTION_UP):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_UP[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)
    
    elif (action == ACTION_STAY):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_STAY[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)
    

    elif(action == ACTION_SHOOT):
        ### success
        new_arrow = arrow -1
        new_health = health - 25
        prob = 1 * SHOOT_PROB_SUCCESS[position]
        
        new_states.append((position, mat, new_arrow, mm_state, new_health))
        probability.append(prob)

        ### failure
        new_arrow = arrow - 1
        new_health = health
        prob = 1 * (1-SHOOT_PROB_SUCCESS[position])
        
        new_states.append((position, mat, new_arrow, mm_state, new_health))
        probability.append(prob)          
    
    elif(action == ACTION_HIT):
        new_states.append((position,mat, arrow, mm_state, max(0, health-50)))
        probability.append(HIT_PROB_SUCCESS[position])

        new_states.append((position,mat, arrow, mm_state, health))
        probability.append(1 - HIT_PROB_SUCCESS[position])
        

    elif(action == ACTION_CRAFT):
        prob = 0
        prob_val = [0.5, 0.35, 0.15]
        
        for i in range(arrow+1,3):
            prob += prob_val[i-1-arrow]
            new_states.append((position, mat-1, i, mm_state, health))
            probability.append(prob_val[i-1-arrow])

        new_states.append((position, mat-1, 3, mm_state, health))
        probability.append(1-prob)


    elif(action == ACTION_GATHER):

        if mat == 2:
            new_states.append((position, 2, arrow, mm_state, health))
            probability.append(1)
        else:
            ## new_states.append((position, max(mat+1,2), arrow, mm_state, health))
            new_states.append((position, mat+1, arrow, mm_state, health))
            probability.append(0.75)
            new_states.append((position, mat, arrow, mm_state, health))
            probability.append(0.25)

    
    l = len(new_states)
    for i in range(0,l):
        rewards.append(0)

    for i in range(0, l):
        if mm_state == STATE_DORMANT:
            probability.append(probability[i]*0.2)
            probability[i] *= 0.8
            new_states.append((new_states[i][0], new_states[i][1], new_states[i][2], STATE_READY, new_states[i][4]))
            rewards.append(rewards[i])
        
        else:
            probability[i] *= 0.5 # Not attacking

            # Attacking but no effect except change in mm_state
            if position != POSITION_C and position != POSITION_E:
                probability.append(probability[i])
                new_states.append((new_states[i][0], new_states[i][1], new_states[i][2], STATE_DORMANT, new_states[i][4]))
                rewards.append(rewards[i])

    # Attack and state becomes original (i.e, IJ is unsuccessful)
    if mm_state == STATE_READY and (position == POSITION_C or position == POSITION_E):
        probability.append(0.5)
        new_states.append((position, mat, 0, STATE_DORMANT, min(health+25, 100)))
        rewards.append(-40)

    return probability, new_states, rewards





## row num will be aligned with ALL_STATES
## col num will be aligned with STATE_ACTIONS
def generate_A(ALL_STATES, STATE_ACTIONS):

    A = np.zeros((len(ALL_STATES), len(STATE_ACTIONS)))

    for state_action in STATE_ACTIONS:

        state, action = state_action
        probability, new_states, rewards = transformation_lp(state, action)
        state_row  = ALL_STATES.index(state)
        state_action_col = STATE_ACTIONS.index(state_action)

        ## Noop always make 1, do not go to loop
        if action == ACTION_NONE:
            A[state_row][state_action_col] = +1
            continue

        ## if not noop
        A[state_row][state_action_col] = 1
        for i in range(0, len(new_states)):
            new_state_row = ALL_STATES.index(new_states[i])
            A[new_state_row][state_action_col] -= probability[i]

    return A



def generate_r(STATE_ACTIONS):
    r = np.zeros((len(STATE_ACTIONS),1))

    for i in range(0, len(STATE_ACTIONS)):
        state, action = STATE_ACTIONS[i]

        ## action noop has no rewards
        if action == ACTION_NONE:
            continue

        probability, new_states, rewards = transformation_lp(state, action)
        # r[i] = step_cost
        for j in range(0, len(new_states)):
            r[i] += probability[j] * (rewards[j] + step_cost)

    return r


def generate_alpha(ALL_STATES):
    alpha = np.zeros((len(ALL_STATES),1))
    i = ALL_STATES.index(((POSITION_C,2,3,STATE_READY,100)))
    alpha[i] = 1
    return alpha


def create_json(A,r, alpha, x, objective, best_actions, ALL_STATES):
    a = list(A)
    a = [list(i) for i in a]

    r = list(r[0])
    x = list(x[0])
    alpha = list(alpha[0])

    policy = []
    for i in range(0,len(ALL_STATES)):
        plan = []
        plan.append(list(ALL_STATES[i]))
        plan.append(best_actions[i][0])

        policy.append(plan)


    result_dict = { "a":a,
                    "r":r,
                    "alpha":alpha,
                    "x":x,
                    "policy":policy,
                    "objective":objective,
    }

    out_file = open("outputs/part_3_output.json", "w")
    json.dump(result_dict, out_file, indent = 6)
    out_file.close()

#######################################################################################################

ALL_STATES = []
for i in POSITIONS:
    for j in MATS:
        for k in ARROWS:
            for l in MM_STATES:
                for m in HEALTHS:
                    state = (i,j,k,l,m)
                    ALL_STATES.append(state)

STATE_ACTIONS = []

for state in ALL_STATES:
    for action in ACTIONS:
        if(isValidAction(state, action)):
            STATE_ACTIONS.append((state, action))

A = generate_A(ALL_STATES, STATE_ACTIONS)
r = generate_r(STATE_ACTIONS)
alpha = generate_alpha(ALL_STATES)


x = cp.Variable(r.shape, 'x',nonneg=True)
constraints = [
    cp.matmul(A, x) == alpha
]

objective = cp.Maximize(cp.matmul(np.transpose(r), x))
problem = cp.Problem(objective, constraints)
solution = problem.solve()


##########################################
best_actions = []
for state in ALL_STATES:
    max_x_value = -100000000
    max_action = ''
    for action in ACTIONS:
        if(isValidAction(state, action)):
            index = STATE_ACTIONS.index((state, action))
            x_value = x.value[index][0]

            if(x_value > max_x_value):
                max_x_value = x_value
                max_action = action
    best_actions.append((max_action, max_x_value))


create_json(A,np.transpose(r),np.transpose(alpha),np.transpose(x.value),solution,best_actions,ALL_STATES)