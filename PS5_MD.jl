#####   PS 5   ##### 
using Parameters, Interpolations, Distributions, Random, Optim, GLM, DataFrames, StatsModels

@with_kw struct Primitives 
    β::Float64 = 0.99  #discount factor
    α::Float64 = 0.36  #capital share
    δ::Float64 = 0.025  #depreciation rate
    z_t::Array{Float64, 1} = [1.01, 0.99]  #aggregate technology shocks [z_g, z_b]
    nz::Int64 = length(z_t) 
    ϵ_t::Array{Float64, 1} = [0, 1]  #employment states  [unemployed, employed]
    k_t::Array{Float64, 1} = collect(range(0.001, stop = 20.0, length = 21))  #capital grid for individuals
    nk::Int64 = length(k_t)
    K_t::Array{Float64, 1} = collect(range(10.0, stop = 15.0, length = 11))   #aggregate capital grid
    nK::Int64 = length(K_t)
    T::Int64 = 11000   #the sequence of productivity shocks z_t
    N::Int64 = 5000  #employment shocks 
    #Below we create the transition matrix
    u_g::Float64 = 0.04   #the fraction of those unemployed in the good state
    u_b::Float64 = 0.10   #the fraction of those unemployed in the bad state
    u_t::Array{Float64, 1} = [u_g, u_b]   #unemployment vector 
    e::Float64 = 0.3271   #labor efficiency per unit of time worked
    e_t::Array{Float64, 1} = [e, 0.0]  #labor efficiency vector
    ne::Int64 = length(e_t)
    d::Float64 = 8.0      #the average duration of good and bad times
    d_g::Float64 = 1.5  #the average duration of unemployment in good times 
    d_b::Float64 = 2.5  #the average duration of unemployment in bad times
    gg00::Float64 = (d_g - 1)/d_g   #the probabibility of going from (good, unemployed) to (good, unemployed)
    bb00::Float64 = (d_b - 1)/d_b   # probability of going from (bad, unemployed) to (bad, unemployed)
    bg00::Float64 = 1.25*bb00       #probability of going from (bad, unemployed) to (good, unemployed)
    gb00::Float64 = 0.75*gg00       #probabilty of going from (good, unemployed) to (bad, unemployed)
    gg01::Float64 = (u_g - u_g*gg00)/(1 - u_g)    # (good, unemployed) to (good, employed)
    bb01::Float64 = (u_b - u_b*bb00)/(1 - u_b)    # (bad, unemployed) to (bad, employed)
    bg01::Float64 = (u_b - u_g*bg00)/(1 - u_g)    # (bad, unemployed) to (good, employed)
    gb01::Float64 = (u_g - u_b*gb00)/(1 - u_b)    # (good, unemployed) to (bad, employed)
    gg::Float64 = (d - 1)/d 
    gb::Float64 = 1 - (d - 1)/d
    gg10::Float64 = 1 - (d_g - 1)/d_g
    bb10::Float64 = 1 - (d_b - 1)/d_b 
    bg10::Float64 = 1 - 1.25*bb00
    gb10::Float64 = 1 - 0.75*gg00
    gg11::Float64 = 1 - (u_g - u_g*gg00)/(1 - u_g)
    bb11::Float64 = 1 - (u_b - u_b*bb00)/(1 - u_b)
    bg11::Float64 = 1 - (u_b - u_g*bg00)/(1 - u_g)
    gb11::Float64 = 1 - (u_g - u_b*gb00)/(1 - u_b)
    bg::Float64 = 1 - (d - 1)/d 
    bb::Float64 = (d -1)/d
    #transition matrix
    markov::Array{Float64, 2} = [gg*gg11 gb*gb11 gg*gg10 gb*gb10; bg*bg11 bb*bb11 bg*bg10 bb*bb10; gg*gg01 gb*gb01 gg*gg00 gb*gb00; bg*bg01 bb*bb01 bg*bg00 bb*bb00]
    #the first row are the outcomes starting with (good, employed). The second row are the outcomes starting with (bad, employed).
    #the thrid row are the outcomes starting with (good, unemployed). The fourth row is for (bad, unemployed) 
    Mgg::Array{Float64, 2} = [gg11 gg01; gg10 gg00]  #employment transition matrix if z_g yesterday and today
    Mbg::Array{Float64, 2} = [bg11 bg01; bg10 bg00]  
    Mgb::Array{Float64, 2} = [gb11 gb01; gb10 gb00]
    Mbb::Array{Float64, 2} = [bb11 bb01; bb10 bb00]
end

mutable struct Results
    a_0::Float64
    b_0::Float64
    a_1::Float64
    b_1::Float64
    val_func::Array{Float64, 4}
    pol_func::Array{Float64, 4}
    R2::Array{Float64, 1}  #R squared 
    Z::Array{Float64, 1}
    ε::Array{Float64, 2}
    K_agg::Array{Float64}
end

function Initialize()
    prim = Primitives()
    #Initial guesses
    a_0 = 0.095  
    b_0 = 0.085  
    a_1 = 0.999
    b_1 = 0.999
    val_func = zeros(prim.nk, prim.ne, prim.nK, prim.nz)
    pol_func = zeros(prim.nk, prim.ne, prim.nK, prim.nz)
    R2 = zeros(2)  #we'll have an R-squared for the good and bad state of the world
    Z = zeros(prim.T)
    ε = zeros(prim.N, prim.T)
    K_agg = zeros(prim.T - 1000)
    res = Results(a_0, b_0, a_1, b_1, val_func, pol_func, R2, Z, ε, K_agg)
    prim, res #return deliverables
end

function Shocks(prim::Primitives)
    @unpack markov, T, N, Mgg, Mgb, Mbg, Mbb, gg, bb = prim
    
    Random.seed!(12032020)
    dist = Uniform(0, 1)
    
    
    #the vector of productivity shocks
    Z = zeros(T)
    Z[1] = 1  #start with the good state z_g which will be coded as 1
    #the matrix of employment status' given the employment shock
    ε = zeros(N, T)
    ε[:, 1] .= 1

    for t = 2:T #looping over time
        z_shock = rand(dist)  #the productivity shock will be randomly drawn from the uniform distribution 
        if Z[t - 1] == 1 && z_shock < gg   #if the state yesterday is z_g and the z shock is less than or equal to the probability of going from #z_g to z_g, 
            #then we code z today to be equal to the good state z_g = 1
            Z[t] = 1
        elseif Z[t - 1] == 1 && z_shock > gg
            Z[t] = 2  #if yesterday was z_g and the z_shock exceeds the probability of the state being z_g today, then we code for z_b = 2
        elseif Z[t - 1] == 2 && z_shock < bb
            Z[t] = 2
        elseif Z[t - 1] == 2 && z_shock > bb
            Z[t] = 1
        end

        #now we fill in ε
        for i = 1:N #looping over the 5000 employment shocks
            ε_shock = rand(dist) #we draw random employment shocks for the uniform distribution
            if Z[t - 1] == 1 && Z[t] == 1  #if yesterday was the good state and today is also the good state
                p11 = Mgg[1, 1]  #probabibility of going from employed to employed
                p00 = Mgg[2, 2]  #probabibility of going from unemployed to unemployed

                if ε[i, t-1] == 1 && ε_shock < p11
                    ε[i, t] = 1
                elseif ε[i, t-1] == 1 && ε_shock > p11
                    ε[i, t] = 2
                elseif  ε[i, t-1] == 2 && ε_shock < p00
                    ε[i, t] = 2
                elseif ε[i, t-1] == 2 && ε_shock > p00
                    ε[i, t] = 1
                end
            elseif Z[t - 1] == 1 && Z[t] == 2 #if yesterday was the good state and today is the bad state
                p11 = Mgb[1, 1]
                p00 = Mgb[2, 2]

                if ε[i, t-1] == 1 && ε_shock < p11
                    ε[i, t] = 1
                elseif ε[i, t-1] == 1 && ε_shock > p11
                    ε[i, t] = 2
                elseif  ε[i, t-1] == 2 && ε_shock < p00
                    ε[i, t] = 2
                elseif ε[i, t-1] == 2 && ε_shock > p00
                    ε[i, t] = 1
                end
            elseif Z[t-1] == 2 && Z[t] == 1 
                p11 = Mbg[1, 1]
                p00 = Mbg[2, 2]

                if ε[i, t-1] == 1 && ε_shock < p11
                    ε[i, t] = 1
                elseif ε[i, t-1] == 1 && ε_shock > p11
                    ε[i, t] = 2
                elseif  ε[i, t-1] == 2 && ε_shock < p00
                    ε[i, t] = 2
                elseif ε[i, t-1] == 2 && ε_shock > p00
                    ε[i, t] = 1
                end
            elseif Z[t-1] == 2 && Z[t] == 2
                p11 = Mbb[1,1]
                p00 = Mbb[2, 2]
                
                if ε[i, t-1] == 1 && ε_shock < p11
                    ε[i, t] = 1
                elseif ε[i, t-1] == 1 && ε_shock > p11
                    ε[i, t] = 2
                elseif  ε[i, t-1] == 2 && ε_shock < p00
                    ε[i, t] = 2
                elseif ε[i, t-1] == 2 && ε_shock > p00
                    ε[i, t] = 1
                end
            end
        end
    end
    res.Z = Z
    res.ε = ε
    return Z, ε
end #end function

#creating a get index function so we can get an index from our interpolated grids
function get_index(val::Float64, grid::Array{Float64, 1})
    n = length(grid)
    index = 0
    if val <= grid[1]  #if the value is less than or equal to the first grid point, return the index of 1
        index = 1  
    elseif val >= grid[n] #if the val exceeds the largest grid point, just return the index n 
        index = n
    else
        index_upper = findfirst(x-> x > val, grid) #the upper index is the index of the first number that's larger than val
        index_lower = index_upper - 1 
        val_upper, val_lower = grid[index_upper], grid[index_lower]
        index = index_lower + (val - val_lower)/(val_upper - val_lower)
    end
    return index
end


function Bellman(prim::Primitives, res::Results)
    @unpack β, α, δ, nk, k_t, ne, e_t, K_t, nK, nz, z_t, u_g, u_b, markov = prim
    @unpack pol_func, val_func, a_0, a_1, b_0, b_1 = res

    v_next = zeros(prim.nk, prim.ne, prim.nK, prim.nz)
    pol_next  = zeros(prim.nk, prim.ne, prim.nK, prim.nz)

    #Calculating labor for each period
    L_t = [0.96, 0.9]   #L_t = [L_g, L_b]
    
    #Now we interpolate our capital grid and value funciton.
    k_interp = interpolate(k_t, BSpline(Linear()))  #interpolating our individual capital grid
    v_interp = interpolate(val_func, BSpline(Linear()))  #interpolating our value function

    for (i_z, z_today) in enumerate(z_t)  #this goes into the z_t grid and calls index i_z and value z_today which corresponds with that index
        for (i_K, K_today) in enumerate(K_t)   #finding the index and value of aggregate capital today
            w_today = (1 - α)*z_today*((K_today/L_t[i_z])^α)  #calculating the (state dependent) wage rate
            r_today = α*z_today*((K_today/L_t[i_z])^(α - 1))  #calculating the (stae dependent) interest rate
            
            #Determining the equation we'll use for capital tomorrow
            if i_z == 1  #if we're in the good state
                K_tom = a_0 + a_1*log(K_today)
            elseif i_z == 2 #if we're in the bad state
                K_tom = b_0 + b_1*log(K_today)
            end
            K_tom = exp(K_tom)

            i_Kp = get_index(K_tom, K_t) #using the get index function from above to get an index of capital tomorrow which will
            #likely be between two of our grid points

            for (i_e, e_today) in enumerate(e_t)  #getting the index and value of labor efficiency 
                row = i_e + ne*(i_z - 1) #getting the row index for our markov matrix. We go to the rows where you are employed today (rows 1 & 2) first
                #then to the rows where you are unemployed today (rows 3 & 4)

                for (i_k, k_today) in enumerate(k_t)  #finding the index and value of individual capital today
                    budget = r_today*k_today + w_today*e_today + (1 - δ)*k_today
                        #Finding the continuation value by interpolating over k and K 
                        v_tom(i_kp) = markov[row,1]*v_interp(i_kp, 1, i_Kp, 1) + markov[row, 2]*v_interp(i_kp, 1, i_Kp, 2) + markov[row, 3]*v_interp(i_kp, 2, i_Kp, 1) + markov[row, 4]*v_interp(i_kp, 1, i_Kp, 2)

                        #Solving the household's problem for k'. 
                        val_f(i_kp) = log(budget - k_interp(i_kp)) + β*v_tom(i_kp)  #agent's value function as a function of their capital choice
                        obj(i_kp) = -val_f(i_kp) #maximize by minimizing the negative
                        lower = 1.0
                        upper = get_index(budget, k_t)
                        opt = optimize(obj, lower, upper) #optimize our value function

                        k_tomorrow = k_interp(opt.minimizer[1])
                        v_today = -opt.minimum

                        #update the policy functions
                        pol_next[i_k, i_e, i_K, i_z] = k_tomorrow
                        pol_func[i_k, i_e, i_K, i_z] = k_tomorrow  #updating the policy function in the results struct
                        v_next[i_k, i_e, i_K, i_z] = v_today
                end
            end
        end
    end

    return v_next
end #end the Bellman 

#Now we iterate our value function until it converges
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        err = abs.(maximum(v_next.-res.val_func))/abs(v_next[prim.nk, prim.ne, prim.nK, prim.nz]) 
        res.val_func = v_next #update value function
        n+=1
    end
    println("Value function converged in ", n, " iterations.")
end


### Step 5: use the decision rules from pol_func and the ε matrix to simulate the savings behavior of N=5000 households starting from the initial condition
# K_ss = 11.55, discarding the first 1000 periods. You'll generate a NxT' matrix where each row is a different agent's k' choice in state (e, z) and T' = T - 1000.

function V_func(prim::Primitives, res::Results)
    @unpack ε, pol_func, K_agg, Z = res
    @unpack N, T, k_t, K_t = prim

    #interpolate the policy function
    pol_func_interpolate = interpolate(pol_func, BSpline(Linear()))

    #At t = 0, we'e in the steady state where aggregate capital K_ss = 11.55. 
    #1. Each household has some initial level of capital
    #2. For household n in period t, go into the ε matrix and figure out what their employment shock is (1 if employed, 2 if unemployed)
    #3. Go into the pol_func and find the index of the aggregate state today which comes from Z[t] where t is the time period you're in currently (t in 1:T')
    #then for each agent, find the k' recommended to them by the policy function given their k today and their employment state given by ε
    V = zeros(N,T)
    k_0 = 11.55/N   
    V[:, 1000] .= k_0 #each household's initial k 
    K_aggregate = zeros(T)
    K_aggregate[1000] = 11.55

    for t = 1001:T #looping over the time periods
        for n = 1:N #looping over households
            k = V[n, t-1] #to get k today, you need to find what your capital holding was yesterday
            e = ε[n, t-1]  #for household n, figure out what their employment status is in time period t-1
            K = K_aggregate[t-1]  #find aggregate capital yesterday
            z = Z[t-1]  #find the state of the world yesterday
            V[n, t] = pol_func_interpolate[get_index(k, k_t), e, get_index(K, K_t), z]  #go to the policy function and see what it tells you k' should be 
        end
        K_aggregate[t] = sum(V[:, t])
    end
    return K_aggregate
end #end function

#6.  Use the simulated data to re-estimate the parameters 
function auto_regression(prim::Primitives, res::Results)
    @unpack K_agg, Z, a_0, a_1, b_0, b_1 = res
    @unpack N, T = prim
    K = zeros(length(K_agg))
    for i = 1:length(K)
        K[i] = K_agg[i]/N   #averaging over all agents in each period
    end
    K   #this is the vector K that we need to estimate the parameters of the log-linear specification

    log_Kp = zeros(10000) #vector of Kp for all states
    for t = 2:10000 #looping over K
        log_Kp[t] = log(K[t])
    end

    log_Kpg = log_Kp[findall(x-> x == 1.0, Z[1001:11000])]  #the is the vector Kp associated with the good state only
    log_Kpb = log_Kp[findall(x-> x == 2.0, Z[1001:11000])]  #the is the vector Kp associated with the bad state only

    #To run the regression, I need to separate the K vector into the good and bad states as well
    #first, I log all of the capital values
    log_K = zeros(length(K_agg))
    for i = 1:length(K)-1
        log_K[i] = log(K_agg[i]/N)   #averagig over all agents in each period
    end
    log_K 

    log_Kg = log_K[findall(x-> x == 1.0, Z[1001:11000])]  #the vector of K associated with the good state 
    log_Kb = log_K[findall(x-> x == 2.0, Z[1001:11000])]  #the vector of K associated with the bad state 

    #running the auto-regressions
    # for the good state
    data_g = DataFrame(X = log_Kg, Y = log_Kpg)
    ols_g = lm(@formula(Y ~ X), data_g)
    a_0_new = coef(ols_g)[1]
    a_1_new = coef(ols_g)[2]

    #for the bad state
    data_b = DataFrame(X = log_Kb, Y = log_Kpb)
    ols_b = lm(@formula(Y ~ X), data_b)
    b_0_new = coef(ols_b)[1]
    b_1_new = coef(ols_b)[2]
    
    #Getting the R2
    ssr_a = sum((log_Kpg .- predict(ols_g)).^2) #sum squared regression in good state
    sst_a = sum((log_Kpg .- mean(log_Kpg)).^2) #total sum of squares in good state 
    ssr_b = sum((log_Kpb .- predict(ols_b)).^2) #sum squared regression in bad state
    sst_b = sum((log_Kpb .- mean(log_Kpb)).^2) #total sum of squares in bad state

    R2 = 1 - (ssr_a + ssr_b)/(sst_a + sst_b)

    return a_0_new, a_1_new, b_0_new, b_1_new, R2
end


function Solve_model(prim::Primitives, res::Results)
    Shocks(prim)
    res.Z = Shocks(prim)[1]  #updating the aggregate shocks
    res.ε = Shocks(prim)[2]  #updating the employment shocks
    Bellman(prim, res)
    V_iterate(prim, res)
    V_func(prim, res)
    res.K_agg = V_func(prim, res)[1001:prim.T]  #updating the aggregate capital vector 
    auto_regression(prim, res)
end

#7. Now, we iterate until the coefficients converge
prim, res = Initialize()
function Iterate(prim::Primitives, res::Results; tol::Float64 = 0.001)
    @unpack a_0, a_1, b_0, b_1 = res  #unpack initial guesses of the coefficients
    prim, res = Initialize()
    #Solve the model once to get new coefficients
    Solve_model(prim, res)
    new_a_0 = auto_regression(prim, res)[1]
    new_a_1 = auto_regression(prim, res)[2]
    new_b_0 = auto_regression(prim, res)[3]
    new_b_1 = auto_regression(prim, res)[4]
    new_R2 = auto_regression(prim, res)[5]

    #calculate the errors for the coefficients
    err_a_0 = new_a_0 - res.a_0
    err_a_1 = new_a_1 - res.a_1
    err_b_0 = new_b_0 - res.b_0
    err_b_1 = new_b_1 - res.b_1
    n = 0 #counter

    while n < 25  #limiting the number of iterations so Julia doesn't iterate forever
        println("on iteration ", n)
        if abs(err_a_0) > tol || abs(err_a_1) > tol || abs(err_b_0) > tol || abs(err_b_1) > tol  #if any of the errors is greater than tol
            res.a_0 = 0.5*res.a_0 + 0.5*new_a_0  #updating the coefficients
            res.a_1 = 0.5*res.a_1 + 0.5*new_a_1 
            res.b_0 = 0.5*res.b_0 + 0.5*new_b_0
            res.b_1 = 0.5*res.b_1 + 0.5*new_b_1

            #solve the model again and get the new coefficient estimates
            Solve_model(prim, res)
            new_a_0 = auto_regression(prim, res)[1]
            new_a_1 = auto_regression(prim, res)[2]
            new_b_0 = auto_regression(prim, res)[3]
            new_b_1 = auto_regression(prim, res)[4]
            new_R2 = auto_regression(prim, res)[5]

            #update the errors
            err_a_0 = new_a_0 - res.a_0
            err_a_1 = new_a_1 - res.a_1
            err_b_0 = new_b_0 - res.b_0
            err_b_1 = new_b_1 - res.b_1

            println("new a_0 ", new_a_0)
            println("new a_1 ", new_a_1)
            println("new b_0 ", new_b_0)
            println("new b_1 ", new_b_1)
            println("new R2  ", new_R2)

        elseif abs(err_a_0) < tol && abs(err_a_1) < tol && abs(err_b_0) < tol && abs(err_b_1) < tol  #if all of the coefficients have converged
            println("all done!")
            println("new a_0 ", new_a_0)
            println("new a_1 ", new_a_1)
            println("new b_0 ", new_b_0)
            println("new b_1 ", new_b_1)
            println("new R2  ", new_R2)
        end
        n+=1
    end #end the while loop 
end #end the function 

Iterate(prim, res)

#It converged after 6 iterations. The final coefficients are as follows
#a_0 = -2.92584e-14
#a_1 = 1.0
#b_0 = 0.00724
#b_1 = 0.9962
#R2 = 0.9983
