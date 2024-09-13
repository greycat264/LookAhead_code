# Features
CREATOR_ATTRIBUTE_FEATURES = [
    "nonce",
    "fund_from_label",
]
TRANSACTION_DATA_FEATURES = ["value", "txDataLen", "gasUsed"]
VERIFICATION_FEATURES = [
    "verity",
]
IMPLEMENTED_FUNC_FEATURES = [
    "publicFucNumber",
    "internalFucNumber",
    "totalFuncNumber",
    "publicFucFrequency",
    "flashloanFuncNumber",
    "FlashloanProportion",
]
FUNC_CALL_FEATURES = [
    "totalCallNumber",
    "totalJmpNumber",
    "totalLinkNumber",
    "tokenCallNumber",
    "tokenCallFrequency",
    "transferCallNumber",
    "balanceOfCallNumber",
    "approveCallNumber",
    "swapCallNumber",
    "totalSupplyCallNumber",
    "allowanceCallNumber",
    "transferFromCallNumber",
    "mintCallNumber",
    "burnCallNumber",
    "withdrawCallNumber",
    "depositCallNumber",
    "skimCallNumber",
    "syncCallNumber",
    "token0CallNumber",
    "token1CallNumber",
    "getReservesCallNumber",
    "avg_token_call",
    "max_token_call",
    "delegateCallNumber",
]
CONTRACT_TYPE_FEATURES = [
    "selfdestructFuncTag",
    "isErcContract",
    "isDelegate",
]
# Will be populated by transformer inference
FUNC_CFA_FEATURES = [
    "callFlowAnalysisConfidence",
]

# Feature sets
TRANSACTION_FEATURES = (
    CREATOR_ATTRIBUTE_FEATURES + TRANSACTION_DATA_FEATURES + VERIFICATION_FEATURES
)
CONTRACT_FEATURES = (
    IMPLEMENTED_FUNC_FEATURES + FUNC_CALL_FEATURES + CONTRACT_TYPE_FEATURES
)
COMBINED_FEATURES = TRANSACTION_FEATURES + CONTRACT_FEATURES
ALL_FEATURES = TRANSACTION_FEATURES + CONTRACT_FEATURES + FUNC_CFA_FEATURES
